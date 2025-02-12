import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split

class PINN(nn.Module):
    def __init__(self, num_hidden_layers=4, num_neurons=64,
                 t_mean=None, t_std=None, z_mean=None, z_std=None, theta_mean=None, theta_std=None):
        super(PINN, self).__init__()
        
        # Store the normalization parameters
        self.t_mean = t_mean
        self.t_std = t_std
        self.z_mean = z_mean
        self.z_std = z_std
        self.theta_mean = theta_mean
        self.theta_std = theta_std

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
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, z, theta):
        # Normalize inputs
        t_norm = normalize(t, self.t_mean, self.t_std)
        z_norm = normalize(z, self.z_mean, self.z_std)
        theta_norm = normalize(theta, self.theta_mean, self.theta_std)

        # Concatenate inputs
        x = torch.cat([t_norm, z_norm, theta_norm], dim=1)  # shape: (batch_size, 3)
        
        # Forward pass
        x = self.input_layer(x)
        x = self.activation(x)
        for hl in self.hidden_layers:
            x = hl(x)
            x = self.activation(x)
        x = self.output_layer(x)
        
        return x  # shape: (batch_size, 1)
    
# ---------------------------------------------------
#                   Thermal properties
# ---------------------------------------------------
def alpha_sand(theta):
    return 0.25 + 0.64 / (1.0 + torch.exp(-1.72*(theta - 6.01)))

def alpha_topsoil(theta):
    return 0.23 + 0.25 / (1.0 + torch.exp(-0.78*(theta - 11.3)))

def C_sand(theta):
    num = 52.7 + 8.32*(theta - 5.27)
    den = 1.0 + torch.exp(-3.24*(theta - 5.27))
    return 296.4 + num/den

def alpha(z, theta, z_top=0.150, beta=100.0):
    # sigma: a smooth approximation to a step function.
    sigma = torch.sigmoid(beta*(z - z_top))
    return alpha_topsoil(theta) * (1 - sigma) + alpha_sand(theta) * sigma

# ---------------------------------------------------
#                   PDE Residual
# ---------------------------------------------------
def pde_residual(model, t, z, theta):
    """
    PDE:  dT/dt = alpha(theta) * d2T/dz^2 + (dalpha/dz)*dT/dz
    """
    # Ensure t and z require gradients
    t.requires_grad = True
    z.requires_grad = True

    # Evaluate the model: T = T(t, z, theta)
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

    # 3) d2T_dz2
    d2T_dz2 = torch.autograd.grad(
        dT_dz,
        z,
        grad_outputs=torch.ones_like(dT_dz),
        retain_graph=True,
        create_graph=True
    )[0]

    # 4) Evaluate alpha (thermal diffusivity)
    alpha_val = alpha(z, theta)

    # 5) Compute dalpha/dz
    dalpha_dz = torch.autograd.grad(
        alpha_val,
        z,
        grad_outputs=torch.ones_like(alpha_val),
        retain_graph=True,
        create_graph=True
    )[0]

    # 6) PDE residual
    residual = dT_dt - alpha_val * d2T_dz2 - dalpha_dz * dT_dz
    return residual


# ---------------------------------------------------
#                  Boundary Conditions
# ---------------------------------------------------
def top_boundary_loss(model, t_bc, z_top, T_bc, theta_bc):
    """
    T(t, z=z_top, theta) = T_bc
    """
    T_pred = model(t_bc, z_top, theta_bc)
    return (T_pred.squeeze() - T_bc.squeeze())**2

def bottom_flux_loss(model, t_bc, z_bot, q_bc, theta_bot, k_bot):
    """
    -k(theta) * dT/dz = q_bc(t) at z=z_bot
    """
    t_bc.requires_grad = True
    z_bot.requires_grad = True

    T_pred = model(t_bc, z_bot, theta_bot)
    
    # dT/dz at bottom
    dT_dz = torch.autograd.grad(T_pred,
                                z_bot,
                                grad_outputs=torch.ones_like(T_pred),
                                retain_graph=True,
                                create_graph=True)[0]
    
    # Use known k_bot directly
    flux_pred = -k_bot * dT_dz
    return (flux_pred.squeeze() - q_bc.squeeze())**2

def initial_condition_loss(model, z_ic, T_ic, theta_ic):
    """
    T(0, z_ic, theta_ic) = T_ic
    """
    t0 = torch.zeros_like(z_ic)
    T_pred = model(t0, z_ic, theta_ic)
    return (T_pred.squeeze() - T_ic.squeeze())**2


# ---------------------------------------------------
#                   Data loss
# ---------------------------------------------------
def data_loss(model, t, z, theta, T_meas):
    T_pred = model(t, z, theta)
    return torch.mean((T_pred - T_meas)**2)

# ---------------------------------------------------
#                 Normalization utility
# ---------------------------------------------------
def normalize(tensor, tensor_mean, tensor_std):
    return (tensor - tensor_mean) / tensor_std

# ---------------------------------------------------
#       Helper function to compute total losses
# ---------------------------------------------------
def compute_all_losses(model,
                       t_col, z_col, theta_col,
                       t_bc_top, z_bc_top, T_bc_top_val, theta_bc_top,
                       t_bc_bot, z_bc_bot, q_bc_bot_val, theta_bc_bot, k_bc_bot,
                       z_ic_vals, T_ic_vals, theta_ic_vals,
                       t_data, z_data, theta_data, T_measured_data,
                       w_pde, w_bc_top, w_bc_bot, w_ic, w_data):
    """
    Returns individual losses and total combined loss
    """
    # PDE residual
    res_pde = pde_residual(model, t_col, z_col, theta_col)
    loss_pde = torch.mean(res_pde**2)
    
    # Top boundary
    loss_bc_top = torch.mean(
        top_boundary_loss(model, t_bc_top, z_bc_top, T_bc_top_val, theta_bc_top)
    )
    
    # Bottom boundary
    loss_bc_bot = torch.mean(
        bottom_flux_loss(model, t_bc_bot, z_bc_bot, q_bc_bot_val, theta_bc_bot, k_bc_bot)
    )
    
    # Initial condition
    loss_ic = torch.mean(
        initial_condition_loss(model, z_ic_vals, T_ic_vals, theta_ic_vals)
    )
    
    # Data mismatch
    loss_data_val = data_loss(model, t_data, z_data, theta_data, T_measured_data)
    
    # Combined loss
    total_loss = (w_pde * loss_pde +
                  w_bc_top * loss_bc_top +
                  w_bc_bot * loss_bc_bot +
                  w_ic * loss_ic +
                  w_data * loss_data_val)
    
    return loss_pde, loss_bc_top, loss_bc_bot, loss_ic, loss_data_val, total_loss

# ---------------------------------------------------
#                 Main: Two-stage training
# ---------------------------------------------------
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    w_pde = 1.0
    w_bc_top = 10
    w_bc_bot = 10
    w_ic = 1.0
    w_data = 10

    # ---------------------------------------------------
    #         Prepare Collocation Points & Data
    # ---------------------------------------------------
    df = pd.read_csv("data/processed/merged_data_interpolated.csv", parse_dates=["Time"])
    df["time_s"] = (df["Time"] - df["Time"].min()).dt.total_seconds()
    df["z_m"] = df["Height"] / 1000.0

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_test.to_csv("data/processed/test_data.csv", index=False)

    # Tensors for training data
    t_data = torch.tensor(df_train["time_s"].values, dtype=torch.float32).reshape(-1,1)
    z_data = torch.tensor(df_train["z_m"].values, dtype=torch.float32).reshape(-1,1)
    theta_data = torch.tensor(df_train["VWC"].values, dtype=torch.float32).reshape(-1,1)
    T_measured_data = torch.tensor(df_train["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)
    k_measured_data = torch.tensor(df_train["ThermalConductivity"].values, dtype=torch.float32).reshape(-1,1)

    # Tensors for test data
    t_test_data = torch.tensor(df_test["time_s"].values, dtype=torch.float32).reshape(-1,1)
    z_test_data = torch.tensor(df_test["z_m"].values, dtype=torch.float32).reshape(-1,1)
    theta_test_data = torch.tensor(df_test["VWC"].values, dtype=torch.float32).reshape(-1,1)
    T_meas_test_data = torch.tensor(df_test["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)

    # Compute normalization parameters
    t_mean, t_std = t_data.mean(), t_data.std()
    z_mean, z_std = z_data.mean(), z_data.std()
    theta_mean, theta_std = theta_data.mean(), theta_data.std()

    # Collocation grid
    t_min = df["time_s"].min()
    t_max = df["time_s"].max()
    z_min = df["z_m"].min()
    z_max = df["z_m"].max()

    N_t = 100
    N_z = 100
    t_lin = np.linspace(t_min, t_max, N_t)
    z_lin = np.linspace(z_min, z_max, N_z)
    T_mesh, Z_mesh = np.meshgrid(t_lin, z_lin, indexing='xy')

    measured_points = np.column_stack((df["time_s"].values, df["z_m"].values))
    theta_measured = df["VWC"].values
    Theta_mesh = griddata(measured_points, theta_measured, (T_mesh, Z_mesh), method='linear')

    # Fill missing with nearest
    nan_idx = np.isnan(Theta_mesh)
    if np.any(nan_idx):
        Theta_mesh[nan_idx] = griddata(measured_points, theta_measured, (T_mesh, Z_mesh), method='nearest')[nan_idx]

    t_col = torch.tensor(T_mesh.flatten(), dtype=torch.float32).reshape(-1,1)
    z_col = torch.tensor(Z_mesh.flatten(), dtype=torch.float32).reshape(-1,1)
    theta_col = torch.tensor(Theta_mesh.flatten(), dtype=torch.float32).reshape(-1,1)

    # ---------------------------------------------------
    #     Boundary & Initial Conditions
    # ---------------------------------------------------
    # Top boundary
    df_top = df[df["z_m"] == df["z_m"].min()].sort_values(by="time_s")
    t_bc_top = torch.tensor(df_top["time_s"].values, dtype=torch.float32).reshape(-1,1)
    T_bc_top_val = torch.tensor(df_top["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)
    theta_bc_top = torch.tensor(df_top["VWC"].values, dtype=torch.float32).reshape(-1,1)

    # Bottom boundary
    df_hf = pd.read_csv("data/raw/PLEXUS@NGIF_HeatFlux_20190718_20190912.csv", parse_dates=["Time"])
    df_hf["time_s"] = (df_hf["Time"] - df_hf["Time"].min()).dt.total_seconds()
    df_hf = df_hf.sort_values(by="time_s")

    df_bot = df[df["z_m"] == df["z_m"].max()].sort_values(by="time_s")
    t_bc_bot = torch.tensor(df_bot["time_s"].values, dtype=torch.float32).reshape(-1,1)
    q_bc_bot_val = torch.tensor(df_hf["HF.Bottom"].values, dtype=torch.float32).reshape(-1,1)
    theta_bc_bot = torch.tensor(df_bot["VWC"].values, dtype=torch.float32).reshape(-1,1)
    k_bc_bot = torch.tensor(df_bot["ThermalConductivity"].values, dtype=torch.float32).reshape(-1,1)

    # Initial condition
    t0_val = df["time_s"].min()
    df_ic = df[df["time_s"] == t0_val].sort_values(by="z_m")
    z_ic_vals = torch.tensor(df_ic["z_m"].values, dtype=torch.float32).reshape(-1,1)
    T_ic_vals = torch.tensor(df_ic["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)
    theta_ic_vals = torch.tensor(df_ic["VWC"].values, dtype=torch.float32).reshape(-1,1)

    # ---------------------------------------------------
    #           Instantiate Model & Send to GPU
    # ---------------------------------------------------
    pinn_model = PINN(num_hidden_layers=4, num_neurons=64,
                      t_mean=t_mean, t_std=t_std,
                      z_mean=z_mean, z_std=z_std,
                      theta_mean=theta_mean, theta_std=theta_std).to(device)

    # Move data to GPU if available
    t_data = t_data.to(device)
    z_data = z_data.to(device)
    theta_data = theta_data.to(device)
    T_measured_data = T_measured_data.to(device)
    k_measured_data = k_measured_data.to(device)

    t_test_data = t_test_data.to(device)
    z_test_data = z_test_data.to(device)
    theta_test_data = theta_test_data.to(device)
    T_meas_test_data = T_meas_test_data.to(device)

    t_col = t_col.to(device)
    z_col = z_col.to(device)
    theta_col = theta_col.to(device)

    t_bc_top = t_bc_top.to(device)
    T_bc_top_val = T_bc_top_val.to(device)
    theta_bc_top = theta_bc_top.to(device)

    t_bc_bot = t_bc_bot.to(device)
    q_bc_bot_val = q_bc_bot_val.to(device)
    theta_bc_bot = theta_bc_bot.to(device)
    k_bc_bot = k_bc_bot.to(device)

    z_ic_vals = z_ic_vals.to(device)
    T_ic_vals = T_ic_vals.to(device)
    theta_ic_vals = theta_ic_vals.to(device)

    # ---------------------------------------------------
    #    STAGE 1: Train with Adam for initial epochs
    # ---------------------------------------------------
    adam_optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
    adam_epochs = 10000

    for epoch in range(adam_epochs):
        adam_optimizer.zero_grad()

        # Compute all losses
        l_pde, l_bc_top, l_bc_bot, l_ic, l_data, total_loss = compute_all_losses(
            pinn_model,
            t_col, z_col, theta_col,
            t_bc_top, torch.full_like(t_bc_top, z_min).to(device), T_bc_top_val, theta_bc_top,
            t_bc_bot, torch.full_like(t_bc_bot, z_max).to(device), q_bc_bot_val, theta_bc_bot, k_bc_bot,
            z_ic_vals, T_ic_vals, theta_ic_vals,
            t_data, z_data, theta_data, T_measured_data,
            w_pde, w_bc_top, w_bc_bot, w_ic, w_data
        )

        total_loss.backward()
        adam_optimizer.step()

        if epoch % 1000 == 0:
            print(f"[Adam] Epoch {epoch}/{adam_epochs} | "
                  f"Total Loss: {total_loss.item():.6f}, "
                  f"PDE: {l_pde.item():.6f}, BC_top: {l_bc_top.item():.6f}, "
                  f"BC_bot: {l_bc_bot.item():.6f}, IC: {l_ic.item():.6f}, "
                  f"Data: {l_data.item():.6f}")

    # ---------------------------------------------------
    #    STAGE 2: Switch to LBFGS for fine-tuning
    # ---------------------------------------------------
    lbfgs_optimizer = optim.LBFGS(pinn_model.parameters(),
                                  lr=1.0,
                                  max_iter=500,
                                  history_size=50)

    # We define a closure that re-computes the loss
    def closure():
        lbfgs_optimizer.zero_grad()
        l_pde, l_bc_top, l_bc_bot, l_ic, l_data, total_loss = compute_all_losses(
            pinn_model,
            t_col, z_col, theta_col,
            t_bc_top, torch.full_like(t_bc_top, z_min).to(device), T_bc_top_val, theta_bc_top,
            t_bc_bot, torch.full_like(t_bc_bot, z_max).to(device), q_bc_bot_val, theta_bc_bot, k_bc_bot,
            z_ic_vals, T_ic_vals, theta_ic_vals,
            t_data, z_data, theta_data, T_measured_data,
            w_pde, w_bc_top, w_bc_bot, w_ic, w_data
        )
        total_loss.backward()
        return total_loss

    lbfgs_steps = 50  # number of outer loop steps
    for i in range(lbfgs_steps):
        # One "step" of LBFGS can involve multiple internal iterations
        loss_value = lbfgs_optimizer.step(closure)

        if i % 5 == 0:
            print(f"[LBFGS] Step {i}/{lbfgs_steps} | Loss: {loss_value:.6f}")

    # ---------------------------------------------------
    #         Save final model & Evaluate on test
    # ---------------------------------------------------
    torch.save(pinn_model.state_dict(), "pinn_model_2_stage.pth")
    print("Model saved as 'pinn_model.pth'.")

    with torch.no_grad():
        T_pred_test = pinn_model(t_test_data, z_test_data, theta_test_data)
        test_mse = torch.mean((T_pred_test - T_meas_test_data)**2).item()
    print("Test MSE:", test_mse)
