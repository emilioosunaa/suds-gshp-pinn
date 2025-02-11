import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PINN(nn.Module):
    def __init__(self, num_hidden_layers=4, num_neurons=64):
        super(PINN, self).__init__()
        
        # Input layer: (t, z, theta) -> hidden
        self.input_layer = nn.Linear(3, num_neurons)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]
        )
        
        # Output layer: hidden -> T
        self.output_layer = nn.Linear(num_neurons, 1)
        
        # Activation
        self.activation = nn.Tanh()
        
        # Initialize weights (optional, but often helpful)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, z, theta):
        # Concatenate inputs
        x = torch.cat([t, z, theta], dim=1)  # shape: (batch_size, 3)
        
        # Forward pass
        x = self.input_layer(x)
        x = self.activation(x)
        for hl in self.hidden_layers:
            x = hl(x)
            x = self.activation(x)
        x = self.output_layer(x)
        
        return x  # shape: (batch_size, 1)
    
#---------------------------------------------------
#                   Thermal properties
# --------------------------------------------------    
def alpha_sand(theta):
    return 0.25 + 0.64 / (1.0 + torch.exp(-1.72*(theta - 6.01)))


def alpha_topsoil(theta):
    return 0.23 + 0.25 / (1.0 + torch.exp(-0.78*(theta - 11.3)))


def C_sand(theta):
    num = 52.7 + 8.32*(theta - 5.27)
    den = 1.0 + torch.exp(-3.24*(theta - 5.27))
    return 296.4 + num/den

def alpha(z, theta, z_top=0.150):
    alpha_val = torch.where(z <= z_top,
                            alpha_topsoil(theta),
                            alpha_sand(theta))
    return alpha_val

#---------------------------------------------------
#                   PDE Residual
# --------------------------------------------------  
def pde_residual(model, t, z, theta):
    """
    Compute PDE residual for:
    dT/dt - alpha(theta)*d2T/dz^2 - (d alpha/dz)*dT/dz = 0
    """
    # Require gradients w.r.t. t, z
    t.requires_grad = True
    z.requires_grad = True
    theta.requires_grad = True
    
    T_pred = model(t, z, theta)  # shape: (N,1)
    
    # 1) dT/dt
    dT_dt = torch.autograd.grad(T_pred,
                                t,
                                grad_outputs=torch.ones_like(T_pred),
                                retain_graph=True,
                                create_graph=True)[0]
    # 2) dT/dz
    dT_dz = torch.autograd.grad(T_pred,
                                z,
                                grad_outputs=torch.ones_like(T_pred),
                                retain_graph=True,
                                create_graph=True)[0]
    # 3) d2T/dz^2
    d2T_dz2 = torch.autograd.grad(dT_dz,
                                  z,
                                  grad_outputs=torch.ones_like(dT_dz),
                                  retain_graph=True,
                                  create_graph=True)[0]
    
    # Evaluate alpha
    alpha_val = alpha(theta, z)
    
    # 4) partial_alpha/partial z
    # First compute dalpha/dtheta
    dalpha_dtheta = torch.autograd.grad(alpha_val,
                                        theta,
                                        grad_outputs=torch.ones_like(alpha_val),
                                        retain_graph=True,
                                        create_graph=True)[0]
    
    # Then partial_alpha/partial z = (dalpha/dtheta) * (d theta / d z)
    dtheta_dz = torch.autograd.grad(theta,
                                    z,
                                    grad_outputs=torch.ones_like(theta),
                                    retain_graph=True,
                                    create_graph=True)[0]
    if dtheta_dz is None:
        # Means theta is not actually a function of z in the computational graph 
        partial_alpha_partial_z = torch.zeros_like(alpha_val)
    else:
        partial_alpha_partial_z = dalpha_dtheta * dtheta_dz
    
    # PDE residual
    residual = dT_dt - alpha_val*d2T_dz2 - partial_alpha_partial_z*dT_dz
    
    return residual

#---------------------------------------------------
#                   Boundary Conditions
# --------------------------------------------------
def top_boundary_loss(model, t_bc, T_bc):
    """
    model: PINN
    t_bc: times at top boundary
    T_bc: known boundary temperature at z=0
    """
    z0 = torch.zeros_like(t_bc)
    theta_dummy = 10.0*torch.ones_like(t_bc)  # TODO: real theta at top boundary
    T_pred = model(t_bc, z0, theta_dummy)
    return (T_pred.squeeze() - T_bc.squeeze())**2

def bottom_flux_loss(model, t_bc, z_bot, q_bc):
    """
    -k(theta) * dT/dz = q_bc(t)
    z_bot: the bottom depth (e.g., 0.94 m)
    q_bc: known flux at bottom boundary
    """
    # TODO: We'll assume a single theta at bottom or a known function
    theta_dummy = 10.0*torch.ones_like(t_bc)
    
    # Forward pass
    t_bc.requires_grad = True
    z_bot.requires_grad = True
    theta_dummy.requires_grad = True
    
    T_pred = model(t_bc, z_bot, theta_dummy)
    
    # dT/dz at bottom
    dT_dz = torch.autograd.grad(T_pred,
                                z_bot,
                                grad_outputs=torch.ones_like(T_pred),
                                retain_graph=True,
                                create_graph=True)[0]
    
    # k = alpha * C
    alpha_val = alpha_sand(theta_dummy)
    C_val = C_sand(theta_dummy)
    k_val = alpha_val * C_val  # (N,1)
    
    # flux_pred = -k * dT_dz
    flux_pred = -k_val * dT_dz
    
    return (flux_pred.squeeze() - q_bc.squeeze())**2

def initial_condition_loss(model, z_ic, T_ic):
    """
    z_ic: depths at time t=0
    T_ic: known initial temp distribution T_init(z)
    """
    t0 = torch.zeros_like(z_ic)
    theta_dummy = 10.0*torch.ones_like(z_ic)
    T_pred = model(t0, z_ic, theta_dummy)
    return (T_pred.squeeze() - T_ic.squeeze())**2

#---------------------------------------------------
#                   Collocation Points
# --------------------------------------------------
# Suppose we have 100 collocation points in time, 100 in space
N_t = 100
N_z = 100

t_min, t_max = 0.0, 10.0   # e.g., 10 days
z_min, z_max = 0.0, 0.94   # 0 to 0.94 m

# Create a mesh
t_lin = np.linspace(t_min, t_max, N_t)
z_lin = np.linspace(z_min, z_max, N_z)
T_mesh, Z_mesh = np.meshgrid(t_lin, z_lin, indexing='xy')

# TODO: For simplicity, let's assume a uniform volumetric water content
# or you have a function to generate real data for theta(t,z)
theta_val = 10.0  # e.g. 10% water content (unit depends)
Theta_mesh = theta_val * np.ones_like(T_mesh)

# Flatten for feeding to the network
t_col = torch.tensor(T_mesh.flatten(), dtype=torch.float32).reshape(-1,1)
z_col = torch.tensor(Z_mesh.flatten(), dtype=torch.float32).reshape(-1,1)
theta_col = torch.tensor(Theta_mesh.flatten(), dtype=torch.float32).reshape(-1,1)

#---------------------------------------------------
#                   Training Loop
# --------------------------------------------------
# Instantiate the model
pinn_model = PINN(num_hidden_layers=4, num_neurons=64)

# Define optimizer
optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)

# Example boundary/initial data (dummy):
t_bc_top = torch.linspace(0, 10, 50).reshape(-1,1)
T_bc_top_val = 20.0 + 2.0*torch.sin(0.5*t_bc_top)  # dummy boundary temp

t_bc_bot = torch.linspace(0, 10, 50).reshape(-1,1)
q_bc_bot_val = 5.0*torch.ones_like(t_bc_bot)  # constant flux as an example

z_ic_vals = torch.linspace(0, 0.94, 50).reshape(-1,1)
T_ic_vals = 15.0*torch.ones_like(z_ic_vals)  # uniform initial temp

# Training hyperparameters
num_epochs = 5000
pde_collocation_batch_size = 2000   # all collocation points or mini-batches

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 1) PDE loss (collocation points)
    res_pde = pde_residual(pinn_model, t_col, z_col, theta_col)
    loss_pde = torch.mean(res_pde**2)
    
    # 2) Top boundary loss
    loss_bc_top = torch.mean(top_boundary_loss(pinn_model, t_bc_top, T_bc_top_val))
    
    # 3) Bottom boundary loss
    z_bot = 0.94*torch.ones_like(t_bc_bot)
    loss_bc_bot = torch.mean(bottom_flux_loss(pinn_model, t_bc_bot, z_bot, q_bc_bot_val))
    
    # 4) Initial condition loss
    loss_ic = torch.mean(initial_condition_loss(pinn_model, z_ic_vals, T_ic_vals))
    
    # Combine losses (tune weights as needed)
    loss = loss_pde + loss_bc_top + loss_bc_bot + loss_ic
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Total Loss: {loss.item():.6f} "
              f"(PDE: {loss_pde.item():.6f}, BC_top: {loss_bc_top.item():.6f}, "
              f"BC_bot: {loss_bc_bot.item():.6f}, IC: {loss_ic.item():.6f})")
