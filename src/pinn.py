import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


# Code based on algorithm 1 from paper https://doi.org/10.1029/2022WR031960
# -----------------------------------
# 1) Define the neural networks
# -----------------------------------
class PsiNet(nn.Module):
    """
    Neural network for the state function ψ(t, z).
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(PsiNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, z):
        """
        Forward pass.
        t, z: Tensors of shape (N,) or broadcastable.
        Returns ψ(t, z).
        """
        # Concatenate inputs (assumes both are 1D, shapes (N,))
        inp = torch.stack([t, z], dim=-1)
        return self.net(inp)


class KNet(nn.Module):
    """
    Neural network for K(ψ).
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(KNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, psi):
        """
        Forward pass for K given ψ.
        psi: Tensor of shape (N, 1).
        """
        return self.net(psi)


class ThetaNet(nn.Module):
    """
    Neural network for θ(ψ).
    """
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(ThetaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, psi):
        """
        Forward pass for θ given ψ.
        psi: Tensor of shape (N, 1).
        """
        return self.net(psi)

# -----------------------------------
# 2) Loss components
# -----------------------------------

def loss_theta(psi_model, theta_model, data_t, data_z, data_theta):
    """
    L_\theta: Data mismatch between predicted θ(ψ(t,z)) and actual data.
    data_t, data_z : Tensors of shape (N,).
    data_theta     : Tensors of shape (N,) ground truth.
    """
    psi_pred = psi_model(data_t, data_z)        # shape (N,1) or (N,)
    # Make sure psi_pred is 2D if needed:
    if psi_pred.dim() == 1:
        psi_pred = psi_pred.unsqueeze(-1)
    theta_pred = theta_model(psi_pred)          # shape (N,1)

    # Mean squared error (example)
    return torch.mean((theta_pred.squeeze() - data_theta)**2)


def loss_rre(psi_model, k_model, theta_model, t_re, z_re):
    """
    L_re: PDE residual at collocation points.
    Here, you must implement the PDE residual for the soil water flow (Richards eq. or similar).
    This is a placeholder showing how one might compute a PDE-based residual.

    t_re, z_re: Collocation points, shape (N,).
    """
    # 1) Compute ψ(t, z)
    psi_pred = psi_model(t_re, z_re)
    # 2) Compute derivatives, e.g. partial derivatives w.r.t. z or t
    #    For example: dψ/dz = ...
    # NOTE: This requires using torch.autograd.grad with create_graph=True
    psi_grad_t = torch.autograd.grad(psi_pred, t_re, 
                                     grad_outputs=torch.ones_like(psi_pred),
                                     retain_graph=True,
                                     create_graph=True)[0]
    psi_grad_z = torch.autograd.grad(psi_pred, z_re,
                                     grad_outputs=torch.ones_like(psi_pred),
                                     retain_graph=True,
                                     create_graph=True)[0]

    # 3) Evaluate K(ψ) and θ(ψ)
    if psi_pred.dim() == 1:
        psi_pred = psi_pred.unsqueeze(-1)
    K_pred = k_model(psi_pred)
    theta_pred = theta_model(psi_pred)

    # 4) PDE residual (placeholder). 
    #    Suppose the PDE is something like: ∂θ/∂t = ∂(K(∂ψ/∂z))/∂z + ...
    #    This is just an illustrative example. 
    #    Replace with the actual PDE for your problem.
    #    residual = dθ/dt - d/dz [ K * dψ/dz ]
    theta_t = torch.autograd.grad(theta_pred, t_re,
                                  grad_outputs=torch.ones_like(theta_pred),
                                  retain_graph=True,
                                  create_graph=True)[0]
    # K * dψ/dz
    flux = K_pred * psi_grad_z
    flux_z = torch.autograd.grad(flux, z_re,
                                 grad_outputs=torch.ones_like(flux),
                                 retain_graph=True,
                                 create_graph=True)[0]
    
    # PDE residual example
    residual = theta_t - flux_z

    # L2 norm of PDE residual
    return torch.mean(residual**2)


def loss_l2_regularization(models, lambda_l2=1e-3):
    """
    L2 regularization on network parameters.
    models: list or tuple of the three models [psi_model, k_model, theta_model].
    """
    l2_sum = 0.0
    for m in models:
        for param in m.parameters():
            l2_sum += torch.sum(param**2)
    return lambda_l2 * l2_sum


# -----------------------------------
# 3) Main training loop
# -----------------------------------

def train_soil_model(D1, D2, maxiter=100000):
    """
    D1 = {(t_i^θ, z_i^θ, θ_i)} for data constraint
    D2 = {(t_i^re, z_i^re)} for PDE residual constraint
    maxiter: integer, maximum number of Adam steps.
    """

    # Unpack data from D1 and D2 as Tensors
    # Suppose D1 = (t_theta, z_theta, theta_observed), shapes all (N_theta,)
    # Suppose D2 = (t_re, z_re), shapes all (N_re,)
    t_theta, z_theta, theta_observed = D1
    t_re, z_re = D2

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_theta, z_theta, theta_observed = t_theta.to(device), z_theta.to(device), theta_observed.to(device)
    t_re, z_re = t_re.to(device), z_re.to(device)

    # Instantiate models
    psi_model = PsiNet().to(device)
    k_model   = KNet().to(device)
    theta_model = ThetaNet().to(device)

    # Define the weights for loss terms
    gamma_theta = 10.0
    gamma_re    = 1.0
    gamma_l2    = 0.001

    # Setup optimizer (Adam) with exponential decay
    # Here lr=1e-3 is an example, adjust as needed
    optimizer = optim.Adam(
        list(psi_model.parameters()) + list(k_model.parameters()) + list(theta_model.parameters()), 
        lr=1e-3
    )
    scheduler = ExponentialLR(optimizer, gamma=0.9999)  # Example decay factor

    # Training loop with Adam
    psi_model.train()
    k_model.train()
    theta_model.train()

    for it in range(maxiter):
        # Zero gradients
        optimizer.zero_grad()

        # Compute sub-losses
        L_theta = loss_theta(psi_model, theta_model, t_theta, z_theta, theta_observed)
        L_re = loss_rre(psi_model, k_model, theta_model, t_re, z_re)
        L_l2 = loss_l2_regularization([psi_model, k_model, theta_model], lambda_l2=gamma_l2)

        # Combine
        L_sm = gamma_theta * L_theta + gamma_re * L_re + L_l2

        # Backprop
        L_sm.backward()
        optimizer.step()
        scheduler.step()  # update learning rate

        # Print progress every 1000 steps (for example)
        if it % 1000 == 0:
            print(f"Iteration {it}/{maxiter}, Loss_sm = {L_sm.item():.6f}, "
                  f"L_theta = {L_theta.item():.6f}, L_re = {L_re.item():.6f}, L_l2 = {L_l2.item():.6f}")

    # -----------------------------------
    # 4) Fine-tune with L-BFGS-B
    # -----------------------------------

    # We can wrap the parameters for the LBFGS-B step. PyTorch's LBFGS does not
    # have a built-in "B" constraints version, but the standard LBFGS class is often
    # used as an approximation. For a true L-BFGS-B, you might need another library
    # or a custom implementation. Below is how you might do with the built-in LBFGS.

    # Switch to LBFGS for final refinement
    lbfgs_optimizer = optim.LBFGS(
        list(psi_model.parameters()) + list(k_model.parameters()) + list(theta_model.parameters()),
        lr=1.0,  # typical LR for LBFGS
        max_iter=500,  # example
        # tolerance_grad=1e-8,
        # tolerance_change=1e-12,
        history_size=100
    )

    def closure():
        lbfgs_optimizer.zero_grad()
        L_theta_ = loss_theta(psi_model, theta_model, t_theta, z_theta, theta_observed)
        L_re_ = loss_rre(psi_model, k_model, theta_model, t_re, z_re)
        L_l2_ = loss_l2_regularization([psi_model, k_model, theta_model], lambda_l2=gamma_l2)
        L_sm_ = gamma_theta * L_theta_ + gamma_re * L_re_ + L_l2_
        L_sm_.backward()
        return L_sm_

    # Perform the LBFGS steps
    lbfgs_optimizer.step(closure)

    # Print final losses
    with torch.no_grad():
        final_L_theta = loss_theta(psi_model, theta_model, t_theta, z_theta, theta_observed).item()
        final_L_re = loss_rre(psi_model, k_model, theta_model, t_re, z_re).item()
        final_L_l2 = loss_l2_regularization([psi_model, k_model, theta_model], lambda_l2=gamma_l2).item()
        final_loss = gamma_theta * final_L_theta + gamma_re * final_L_re + final_L_l2
        print("After LBFGS fine-tuning:")
        print(f"  L_theta = {final_L_theta:.6f}")
        print(f"  L_re    = {final_L_re:.6f}")
        print(f"  L_l2    = {final_L_l2:.6f}")
        print(f"  total   = {final_loss:.6f}")

    return psi_model, k_model, theta_model


# -----------------------------------
# Example Usage
# -----------------------------------
if __name__ == "__main__":
    # Create example synthetic data for D1 and D2
    # Replace with actual data loading in practice
    N_theta = 100
    N_re = 200

    # Synthetic: times and depths in [0,1], random
    t_theta = torch.rand(N_theta)
    z_theta = torch.rand(N_theta)
    # Synthetic "observed" theta
    theta_observed = torch.sin(t_theta) + 0.1*z_theta  # made-up relationship

    # Collocation points
    t_re = torch.rand(N_re)
    z_re = torch.rand(N_re)

    D1 = (t_theta, z_theta, theta_observed)
    D2 = (t_re, z_re)

    # Run training
    psi_model_trained, k_model_trained, theta_model_trained = train_soil_model(D1, D2, maxiter=2000)
