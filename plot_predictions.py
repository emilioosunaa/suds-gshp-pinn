import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from src import heat_transfer_pinn


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
})

# ---------------------------
# 1. Read/prepare data
# ---------------------------
df = pd.read_csv("data/processed/merged_data_interpolated.csv", parse_dates=["Time"])
df["time_s"] = (df["Time"] - df["Time"].min()).dt.total_seconds()
df["z_m"] = df["Height"] / 1000.0

t_data = torch.tensor(df["time_s"].values, dtype=torch.float32).reshape(-1,1)
z_data = torch.tensor(df["z_m"].values, dtype=torch.float32).reshape(-1,1)
theta_data = torch.tensor(df["VWC"].values, dtype=torch.float32).reshape(-1,1)

t_mean, t_std = t_data.mean(), t_data.std()
z_mean, z_std = z_data.mean(), z_data.std()
theta_mean, theta_std = theta_data.mean(), theta_data.std()

df_test = pd.read_csv("data/processed/test_data.csv", parse_dates=["Time"])
df_test["time_s"] = (df_test["Time"] - df_test["Time"].min()).dt.total_seconds()
df_test["z_m"] = df_test["Height"] / 1000.0

t_data_test = torch.tensor(df_test["time_s"].values, dtype=torch.float32).reshape(-1,1)
z_data_test = torch.tensor(df_test["z_m"].values, dtype=torch.float32).reshape(-1,1)
theta_data_test = torch.tensor(df_test["VWC"].values, dtype=torch.float32).reshape(-1,1)
T_measured_data_test = torch.tensor(df_test["SoilTemp"].values, dtype=torch.float32).reshape(-1,1)

# Load your trained PINN model
model_test = heat_transfer_pinn.PINN(num_hidden_layers=4, num_neurons=64,
                                    t_mean=t_mean, t_std=t_std,
                                    z_mean=z_mean, z_std=z_std,
                                    theta_mean=theta_mean, theta_std=theta_std)
model_test.load_state_dict(torch.load("pinn_model_2_stage.pth"))
model_test.eval()

with torch.no_grad():
    T_pred_all = model_test(t_data_test, z_data_test, theta_data_test).squeeze().cpu().numpy()
    T_measured_all = T_measured_data_test.squeeze().cpu().numpy()
    t_all = t_data_test.squeeze().cpu().numpy()

# Convert back to NumPy arrays for easier indexing
df_test["T_pred"] = T_pred_all
df_test["T_measured"] = T_measured_all
df_test["t_s"] = t_all

# ---------------------------
# 2. Define metric functions
# ---------------------------
def nRMSE(y_true, y_pred):
    """
    Normalized Root Mean Squared Error.
    Usually normalized by the mean of y_true (or you could choose another scale).
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse / np.mean(y_true) * 100.0

def MAPE(y_true, y_pred):
    """
    Mean Absolute Percentage Error (in %).
    """
    # Add a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0

# ---------------------------
# 3. Plotting
# ---------------------------
unique_heights = [100, 250, 350, 450, 550, 650, 750, 850]  # mm
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))
axes = axes.flatten()

for i, height_mm in enumerate(unique_heights):
    ax = axes[i]
    
    # Select and sort data for this height
    mask = (df_test["Height"] == height_mm)
    t_plot = df_test.loc[mask, "time_s"].values
    T_pred_plot = df_test.loc[mask, "T_pred"].values
    T_meas_plot = df_test.loc[mask, "T_measured"].values
    
    sort_idx = np.argsort(t_plot)
    t_plot = t_plot[sort_idx]
    T_pred_plot = T_pred_plot[sort_idx]
    T_meas_plot = T_meas_plot[sort_idx]
    
    # Downsample data for plotting (e.g., take every 10th point)
    downsample_factor = 10
    t_plot = t_plot[::downsample_factor]
    T_pred_plot = T_pred_plot[::downsample_factor]
    T_meas_plot = T_meas_plot[::downsample_factor]
    
    # Compute metrics (in % for both nRMSE & MAPE)
    nrmse_val = nRMSE(T_meas_plot, T_pred_plot)
    mape_val = MAPE(T_meas_plot, T_pred_plot)
    
    # Plot measured (small black dots) & predicted (blue line)
    ax.plot(t_plot, T_meas_plot, 'ko', markersize=2, label='Measured')
    ax.plot(t_plot, T_pred_plot, color='red', linewidth=1, linestyle='-', label='Predicted')
    
    # Put the height on the upper-left corner
    ax.text(
        0.02, 0.92,
        f"Height = {height_mm} mm",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10
    )
    # Put the metrics (nRMSE & MAPE) on the upper-right corner
    ax.text(
        0.98, 0.92,
        f"nRMSE={nrmse_val:.1f}%, MAPE={mape_val:.1f}%",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10
    )
    
    # Set a common y-limit & ticks (if desired)
    ax.set_ylim(10, 30)
    ax.set_yticks([15, 20, 25, 30])
    
    # Optionally add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add legend only in the first subplot
    if i == 0:
        ax.legend(loc='best')

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.95, wspace=0.3, hspace=0.3)

# One common X-axis label at bottom center
fig.text(0.5, 0.02, "Time (s)", ha='center')

# One common Y-axis label at middle left
fig.text(0.02, 0.5, "Soil Temperature (°C)", va='center', rotation='vertical')

plt.show()