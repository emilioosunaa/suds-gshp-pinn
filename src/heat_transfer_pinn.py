import numpy as np
import tensorflow as tf

# ------------------------------------------------
# 1. Define the neural network for T(t, z)
# ------------------------------------------------
class PINN(tf.keras.Model):
    def __init__(self, hidden_layers=4, hidden_units=64):
        super().__init__()
        self.hidden_layers = []
        # Build an MLP
        for _ in range(hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units, activation='tanh'))
        self.out_layer = tf.keras.layers.Dense(1, activation=None)  # 1 output => T

    def call(self, inputs):
        # inputs is a 2D tensor of shape [batch_size, 2], representing (t, z)
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)

# ------------------------------------------------
# 2. Helper: compute derivatives using tf.GradientTape
# ------------------------------------------------
def derivatives(model, t, z):
    """
    Returns T, dT/dt, dT/dz, d^2T/dz^2 given a neural network 'model' for T(t,z).
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        tape2.watch(z)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            tape1.watch(z)
            # Forward pass
            T = model(tf.stack([t, z], axis=1))  # shape [N, 1]
        # First derivatives
        dT_dt = tape1.gradient(T, t)
        dT_dz = tape1.gradient(T, z)
    # Second derivative
    d2T_dz2 = tape2.gradient(dT_dz, z)
    return T, dT_dt, dT_dz, d2T_dz2

# ------------------------------------------------
# 3. The known alpha(t,z) function
# ------------------------------------------------
def alpha_topsoil(theta):
    return 0.23 + 0.25/(1 + tf.exp(-0.78*(theta - 11.3)))

def alpha_sand(theta):
    return 0.25 + 0.64/(1 + tf.exp(-1.72*(theta - 6.01)))

def C_sand(theta):
    return (296.4 
            + (52.7 + 8.32*(theta - 5.27))
              / (1 + tf.exp(-3.24*(theta - 5.27))))
def alpha(t, z):


# ------------------------------------------------
# 4. PDE residual function
# ------------------------------------------------
def pde_residual(model, t, z):
    """
    Compute the PDE residual:
      PDE = dT_dt - alpha * d2T_dz2 - (d alpha/dz) * dT_dz
    We approximate partial_alpha/partial_z by automatic differentiation too.
    """
    with tf.GradientTape() as tape_alpha:
        tape_alpha.watch(z)
        alpha_val = alpha_fn(t, z)
    alpha_z = tape_alpha.gradient(alpha_val, z)

    T, dT_dt, dT_dz, d2T_dz2 = derivatives(model, t, z)

    # PDE residual
    pde = dT_dt - alpha_val * d2T_dz2 - alpha_z * dT_dz
    return pde

# ------------------------------------------------
# 5. Boundary condition losses
# ------------------------------------------------
def top_boundary_loss(model, t_top, z_top, T_top):
    """
    For top boundary, we have T(t, z_top) = T_top(t).
    This can be enforced by L2 error between model prediction and boundary data.
    """
    T_pred = model(tf.stack([t_top, z_top], axis=1))
    return tf.reduce_mean((T_pred - T_top)**2)

def bottom_flux_loss(model, t_bot, z_bot, flux_bot):
    """
    If your boundary condition is a heat flux at z_bot:
      q(t) = - k(t,z_bot) * dT/dz(t,z_bot)
    or something involving alpha, thermal conductivity, etc.
    We'll do a simple example where flux_bot(t) = - k * dT/dz(t,z_bot).
    """
    # You might need to define k(t,z) or relate it to alpha(t,z)*C(t,z).
    # For demonstration, let's assume k is a known function or constant
    k_val = 1.0  # placeholder
    with tf.GradientTape() as tape:
        tape.watch(z_bot)
        T_bot = model(tf.stack([t_bot, z_bot], axis=1))
    dTdz_bot = tape.gradient(T_bot, z_bot)
    
    flux_pred = -k_val * dTdz_bot
    return tf.reduce_mean((flux_pred - flux_bot)**2)

def initial_condition_loss(model, t_init, z_init, T_init):
    """
    For t=0: T(0,z) = T_init(z).
    """
    T_pred = model(tf.stack([t_init, z_init], axis=1))
    return tf.reduce_mean((T_pred - T_init)**2)

# ------------------------------------------------
# 6. Overall loss function
# ------------------------------------------------
def total_loss(model, 
               t_pde, z_pde,
               t_top, z_top, T_top,
               t_bot, z_bot, flux_bot,
               t_init, z_init, T_init):
    # PDE residual on collocation points
    pde_res = pde_residual(model, t_pde, z_pde)
    loss_pde = tf.reduce_mean(tf.square(pde_res))

    # Boundary losses
    loss_bc_top = top_boundary_loss(model, t_top, z_top, T_top)
    loss_bc_bot = bottom_flux_loss(model, t_bot, z_bot, flux_bot)

    # Initial condition loss
    loss_ic = initial_condition_loss(model, t_init, z_init, T_init)

    # Weighted sum of losses
    # You can tune these weights
    return loss_pde + loss_bc_top + loss_bc_bot + loss_ic

# ------------------------------------------------
# 7. Training Loop
# ------------------------------------------------
def train_pinn(model, 
               t_pde, z_pde,
               t_top, z_top, T_top,
               t_bot, z_bot, flux_bot,
               t_init, z_init, T_init,
               epochs=5000, lr=1e-3):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss_value = total_loss(model, 
                                    t_pde, z_pde,
                                    t_top, z_top, T_top,
                                    t_bot, z_bot, flux_bot,
                                    t_init, z_init, T_init)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value.numpy():.6f}")

# ------------------------------------------------
# 8. Example Data Setup (Synthetic or from measurements)
# ------------------------------------------------
# Suppose you have:
#   - PDE collocation points (t_pde, z_pde)
#   - Measured or specified boundary conditions at top and bottom
#   - Initial condition data
# These should be NumPy arrays or tf.Tensors.

# For demonstration only:
N_colloc = 1000
t_pde_sample = np.random.rand(N_colloc, 1) * 10.0  # e.g. times from 0 to 10 days
z_pde_sample = np.random.rand(N_colloc, 1) * 1.0   # e.g. depth from 0.0 to 1.0 m
t_pde_tf = tf.constant(t_pde_sample, dtype=tf.float32)
z_pde_tf = tf.constant(z_pde_sample, dtype=tf.float32)

# Top boundary condition (interpolated T_top(t))
# For demonstration, let's just do T_top=some function
t_top_data = np.linspace(0, 10, 50)[:, None]
z_top_data = 0.05 * np.ones_like(t_top_data)  # 0.05 m
T_top_vals = 15.0 + 2.0*np.sin(0.5*t_top_data)  # example
t_top_tf = tf.constant(t_top_data, dtype=tf.float32)
z_top_tf = tf.constant(z_top_data, dtype=tf.float32)
T_top_tf = tf.constant(T_top_vals, dtype=tf.float32)

# Bottom flux (mock data)
t_bot_data = np.linspace(0, 10, 50)[:, None]
z_bot_data = 0.94 * np.ones_like(t_bot_data)  # 0.94 m
flux_bot_vals = -10.0 + 2.0*np.cos(0.3*t_bot_data)  # example flux
t_bot_tf = tf.constant(t_bot_data, dtype=tf.float32)
z_bot_tf = tf.constant(z_bot_data, dtype=tf.float32)
flux_bot_tf = tf.constant(flux_bot_vals, dtype=tf.float32)

# Initial condition T(0,z)
z_init_data = np.linspace(0, 1.0, 50)[:, None]
t_init_data = np.zeros_like(z_init_data)
T_init_vals = 10.0 + 3.0*z_init_data  # example gradient
t_init_tf = tf.constant(t_init_data, dtype=tf.float32)
z_init_tf = tf.constant(z_init_data, dtype=tf.float32)
T_init_tf = tf.constant(T_init_vals, dtype=tf.float32)

# ------------------------------------------------
# 9. Instantiate and train the PINN
# ------------------------------------------------
model = PINN(hidden_layers=4, hidden_units=64)
train_pinn(model, 
           t_pde_tf, z_pde_tf,
           t_top_tf, z_top_tf, T_top_tf,
           t_bot_tf, z_bot_tf, flux_bot_tf,
           t_init_tf, z_init_tf, T_init_tf,
           epochs=2000, lr=1e-3)

# ------------------------------------------------
# 10. After training, predictions:
# ------------------------------------------------
t_test = tf.constant([[5.0]], dtype=tf.float32)   # time = 5
z_test = tf.constant([[0.5]], dtype=tf.float32)   # depth = 0.5
T_pred = model(tf.stack([t_test, z_test], axis=1))
print("Predicted temperature at t=5, z=0.5:", T_pred.numpy())
