import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_pinn_model():
    """Build PINN with explicit input definition"""
    inputs = layers.Input(shape=(3,), name="input_features")  # [z, t, θ]
    x = layers.Dense(128, activation='tanh')(inputs)
    x = layers.Dense(128, activation='tanh')(x)
    x = layers.Dense(64, activation='tanh')(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs=inputs, outputs=outputs)

class PhysicsInformedModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pinn = build_pinn_model()
        
    def train_step(self, data):
        # Input data is already the full batch [z, t, theta]
        inputs = data  # Shape: (batch_size, 3)
        
        # Split into components
        z = inputs[:, 0]  # Depth [mm]
        t = inputs[:, 1]  # Time [hr]
        theta = inputs[:, 2]  # Volumetric water content
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([z, t])
            
            # Predict temperatures
            T_pred = self.pinn(inputs)
            
            # Compute derivatives
            dT_dz = tape.gradient(T_pred, z)
            dT_dt = tape.gradient(T_pred, t)
            d2T_dz2 = tape.gradient(dT_dz, z)
            
            # Thermal properties calculations
            alpha = thermal_diffusivity(z, theta)
            C = heat_capacity(z, theta)
            
            # Compute PDE residual
            with tf.GradientTape() as tape_alpha:
                tape_alpha.watch(z)
                alpha_z = thermal_diffusivity(z, theta)
            dalpha_dz = tape_alpha.gradient(alpha_z, z)
            
            pde_residual = dT_dt - (alpha * d2T_dz2 + dalpha_dz * dT_dz)
            
            # Boundary conditions (example)
            bc_mask = tf.abs(z - 50) < 1e-3
            bc_loss = tf.reduce_mean(tf.square(T_pred[bc_mask] - 20.0))
            
            total_loss = tf.reduce_mean(tf.square(pde_residual)) + bc_loss

        # Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {"loss": total_loss}

# Initialize and compile
model = PhysicsInformedModel()
model.compile(optimizer=keras.optimizers.Adam(0.001))

# Training data - now properly formatted as (batch_size, 3)
train_data = np.array([
    [50.0, 0.0, 0.18],
    [850.0, 0.0, 0.22],
    [500.0, 12.5, 0.15],
], dtype=np.float32)

# Train the model
model.fit(train_data, epochs=10, batch_size=3)