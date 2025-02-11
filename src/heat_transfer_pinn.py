import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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
    PDE:  dT/dt = alpha(theta) * d2T/dz^2
    """
    # We only differentiate T w.r.t. t and z:
    t.requires_grad = True
    z.requires_grad = True
    
    # Theta can be a plain tensor (no 'requires_grad'):
    T_pred = model(t, z, theta)  # shape: (N,1)

    # 1) dT/dt
    dT_dt = torch.autograd.grad(
        T_pred,
        t,
        grad_outputs=torch.ones_like(T_pred),
        retain_graph=True,
        create_graph=True
    )[0]

    # 2) dT/dz
    dT_dz = torch.autograd.grad(
        T_pred,
        z,
        grad_outputs=torch.ones_like(T_pred),
        retain_graph=True,
        create_graph=True
    )[0]

    # 3) d2T/dz^2
    d2T_dz2 = torch.autograd.grad(
        dT_dz,
        z,
        grad_outputs=torch.ones_like(dT_dz),
        retain_graph=True,
        create_graph=True
    )[0]

    # 4) Evaluate alpha from the measured theta
    alpha_val = alpha(z, theta)

    # 5) PDE residual (no partial_alpha_partial_z term)
    residual = dT_dt - alpha_val * d2T_dz2

    return residual


#---------------------------------------------------
#                   Boundary Conditions
# --------------------------------------------------
def top_boundary_loss(model, t_bc, T_bc, theta_bc):
    """
    model: PINN
    t_bc: times at top boundary
    T_bc: known boundary temperature at z=0.40
    theta_bc: known theta at top boundary
    """
    z0 = 0.04 * torch.ones_like(t_bc)
    T_pred = model(t_bc, z0, theta_bc)
    return (T_pred.squeeze() - T_bc.squeeze())**2

def bottom_flux_loss(model, t_bc, z_bot, q_bc, theta_bot):
    """
    -k(theta) * dT/dz = q_bc(t)
    z_bot: the bottom depth (e.g., 0.94 m)
    q_bc: known flux at bottom boundary
    """
    # Forward pass
    t_bc.requires_grad = True
    z_bot.requires_grad = True
    theta_bot.requires_grad = True
    
    T_pred = model(t_bc, z_bot, theta_bot)
    
    # dT/dz at bottom
    dT_dz = torch.autograd.grad(T_pred,
                                z_bot,
                                grad_outputs=torch.ones_like(T_pred),
                                retain_graph=True,
                                create_graph=True)[0]
    
    # k = alpha * C
    alpha_val = alpha_sand(theta_bot)
    C_val = C_sand(theta_bot)
    k_val = alpha_val * C_val  # (N,1)
    
    # flux_pred = -k * dT_dz
    flux_pred = -k_val * dT_dz
    
    return (flux_pred.squeeze() - q_bc.squeeze())**2

def initial_condition_loss(model, z_ic, T_ic, theta_ic):
    """
    z_ic: depths at time t=0
    T_ic: known initial temp distribution T_init(z)
    """
    t0 = torch.zeros_like(z_ic)
    T_pred = model(t0, z_ic, theta_ic)
    return (T_pred.squeeze() - T_ic.squeeze())**2
#---------------------------------------------------
#                   Data loss
# --------------------------------------------------
def data_loss(model, t, z, theta, T_meas):
    T_pred = model(t, z, theta)
    return torch.mean((T_pred - T_meas)**2)

if __name__ == "__main__":
    #---------------------------------------------------
    #         Prepare Collocation Points & Data
    #---------------------------------------------------
    # Load your measured data:
    df = pd.read_csv("data/processed/merged_data_interpolated.csv", parse_dates=["Time"])
    df["time_s"] = (df["Time"] - df["Time"].min()).dt.total_seconds()
    df["z_m"] = df["Height"] / 1000.0

    # Create tensors for the data (for data loss):
    t_data = torch.tensor(df["time_s"].values, dtype=torch.float32).reshape(-1,1)
    z_data = torch.tensor(df["z_m"].values, dtype=torch.float32).reshape(-1,1)
    theta_data = torch.tensor(df["VWC"].values, dtype=torch.float32).reshape(-1,1)
    T_measured_data = torch.tensor(df["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)

    # For PDE collocation, you may still use a mesh covering the domain.
    t_min = df["time_s"].min()
    t_max = df["time_s"].max()
    z_min = df["z_m"].min()
    z_max = df["z_m"].max()

    N_t = 100
    N_z = 100
    t_lin = np.linspace(t_min, t_max, N_t)
    z_lin = np.linspace(z_min, z_max, N_z)
    T_mesh, Z_mesh = np.meshgrid(t_lin, z_lin, indexing='xy')

    # Each measured point is (time, depth)
    measured_points = np.column_stack((df["time_s"].values, df["z_m"].values))
    theta_measured = df["VWC"].values  # measured θ values

    # Interpolate θ on the collocation grid (linear interpolation)
    Theta_mesh = griddata(measured_points, theta_measured, (T_mesh, Z_mesh), method='linear')

    # For points outside the convex hull, fill missing values with nearest-neighbor interpolation:
    nan_idx = np.isnan(Theta_mesh)
    if np.any(nan_idx):
        Theta_mesh[nan_idx] = griddata(measured_points, theta_measured, (T_mesh, Z_mesh), method='nearest')[nan_idx]

    t_col = torch.tensor(T_mesh.flatten(), dtype=torch.float32).reshape(-1,1)
    z_col = torch.tensor(Z_mesh.flatten(), dtype=torch.float32).reshape(-1,1)
    theta_col = torch.tensor(Theta_mesh.flatten(), dtype=torch.float32).reshape(-1,1)

    #---------------------------------------------------
    #       Extract Boundary & Initial Conditions from Data
    #---------------------------------------------------
    # Top boundary: use the shallowest sensor data.
    df_top = df[df["z_m"] == df["z_m"].min()].sort_values(by="time_s")
    t_bc_top = torch.tensor(df_top["time_s"].values, dtype=torch.float32).reshape(-1,1)
    T_bc_top_val = torch.tensor(df_top["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)
    theta_bc_top = torch.tensor(df_top["VWC"].values, dtype=torch.float32).reshape(-1,1)

    # Bottom boundary: use the deepest sensor data.
    df_hf = pd.read_csv("data/raw/PLEXUS@NGIF_HeatFlux_20190718_20190912.csv", parse_dates=["Time"])
    df_hf["time_s"] = (df_hf["Time"] - df_hf["Time"].min()).dt.total_seconds()
    df_hf = df_hf.sort_values(by="time_s")
    df_bot = df[df["z_m"] == df["z_m"].max()].sort_values(by="time_s")
    t_bc_bot = torch.tensor(df_bot["time_s"].values, dtype=torch.float32).reshape(-1,1)
    q_bc_bot_val = torch.tensor(df_hf["HF.Bottom"].values, dtype=torch.float32).reshape(-1, 1)
    theta_bc_bot = torch.tensor(df_bot["VWC"].values, dtype=torch.float32).reshape(-1,1)

    # Initial condition: use data from the earliest time.
    t0_val = df["time_s"].min()
    df_ic = df[df["time_s"] == t0_val].sort_values(by="z_m")
    z_ic_vals = torch.tensor(df_ic["z_m"].values, dtype=torch.float32).reshape(-1,1)
    T_ic_vals = torch.tensor(df_ic["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)
    theta_ic_vals = torch.tensor(df_ic["VWC"].values, dtype=torch.float32).reshape(-1,1)


    #---------------------------------------------------
    #                Training Loop
    #---------------------------------------------------
    pinn_model = PINN(num_hidden_layers=4, num_neurons=64)
    optimizer = optim.Adam(pinn_model.parameters(), lr=3e-4)

    num_epochs = 1000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 1) PDE residual loss (from collocation points)
        res_pde = pde_residual(pinn_model, t_col, z_col, theta_col)
        loss_pde = torch.mean(res_pde**2)
        
        # 2) Top boundary loss (using measured top-boundary theta)
        loss_bc_top = torch.mean(top_boundary_loss(pinn_model, t_bc_top, T_bc_top_val, theta_bc_top))
        
        # 3) Bottom boundary loss (using measured bottom-boundary theta)
        z_bot = torch.full_like(t_bc_bot, z_max)  # set bottom depth to max z in domain
        loss_bc_bot = torch.mean(bottom_flux_loss(pinn_model, t_bc_bot, z_bot, q_bc_bot_val, theta_bc_bot))
        
        # 4) Initial condition loss (using measured theta at t=0)
        loss_ic = torch.mean(initial_condition_loss(pinn_model, z_ic_vals, T_ic_vals, theta_ic_vals))
        
        # 5) Data mismatch loss (comparing model predictions to all measured data)
        loss_data = data_loss(pinn_model, t_data, z_data, theta_data, T_measured_data)
        
        # Combine losses (you may want to weight these differently)
        loss = 100000*loss_pde + loss_bc_top + 0.1*loss_bc_bot + loss_ic + loss_data
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Total Loss: {loss.item():.6f} "
                f"(PDE: {loss_pde.item():.6f}, BC_top: {loss_bc_top.item():.6f}, "
                f"BC_bot: {loss_bc_bot.item():.6f}, IC: {loss_ic.item():.6f}, Data: {loss_data.item():.6f})")

    torch.save(pinn_model.state_dict(), "pinn_model.pth")
    print("Model saved as 'pinn_model.pth'.")