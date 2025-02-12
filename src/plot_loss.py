import re
import matplotlib.pyplot as plt

# Enable LaTeX rendering for better formatting (optional)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
})

raw_log = """
[Adam] Epoch 0/10000 | Total Loss: 7039.457031, PDE: 0.211192, BC_top: 270.980011, BC_bot: 63.456860, IC: 307.278320, Data: 338.759857
[Adam] Epoch 1000/10000 | Total Loss: 147.911133, PDE: 9.640023, BC_top: 2.826586, BC_bot: 7.116752, IC: 1.970003, Data: 3.686773
[Adam] Epoch 2000/10000 | Total Loss: 91.295456, PDE: 4.145013, BC_top: 1.841923, BC_bot: 5.249465, IC: 1.360901, Data: 1.487567
[Adam] Epoch 3000/10000 | Total Loss: 78.335159, PDE: 2.771993, BC_top: 1.637200, BC_bot: 4.840578, IC: 0.310585, Data: 1.047480
[Adam] Epoch 4000/10000 | Total Loss: 73.145592, PDE: 2.303039, BC_top: 1.556760, BC_bot: 4.607909, IC: 0.229895, Data: 0.896597
[Adam] Epoch 5000/10000 | Total Loss: 69.463150, PDE: 2.300321, BC_top: 1.510697, BC_bot: 4.359811, IC: 0.189078, Data: 0.826867
[Adam] Epoch 6000/10000 | Total Loss: 74.200386, PDE: 2.290279, BC_top: 1.447104, BC_bot: 4.901252, IC: 0.215076, Data: 0.821147
[Adam] Epoch 7000/10000 | Total Loss: 65.895599, PDE: 2.087703, BC_top: 1.418495, BC_bot: 4.227591, IC: 0.138698, Data: 0.720834
[Adam] Epoch 8000/10000 | Total Loss: 62.008690, PDE: 1.964530, BC_top: 1.362593, BC_bot: 3.933784, IC: 0.132905, Data: 0.694749
[Adam] Epoch 9000/10000 | Total Loss: 65.676392, PDE: 2.263226, BC_top: 1.344875, BC_bot: 4.312306, IC: 0.129260, Data: 0.671209
[LBFGS] Step 0/100 | Loss: 60.608932
[LBFGS] Step 5/100 | Loss: 29.775864
[LBFGS] Step 10/100 | Loss: 14.582275
[LBFGS] Step 15/100 | Loss: 8.746788
[LBFGS] Step 20/100 | Loss: 5.782499
[LBFGS] Step 25/100 | Loss: 4.255418
[LBFGS] Step 30/100 | Loss: 3.466590
[LBFGS] Step 35/100 | Loss: 2.968884
[LBFGS] Step 40/100 | Loss: 2.724864
[LBFGS] Step 45/100 | Loss: 2.537281
[LBFGS] Step 50/100 | Loss: 2.389547
[LBFGS] Step 55/100 | Loss: 2.226818
[LBFGS] Step 60/100 | Loss: 2.074207
[LBFGS] Step 65/100 | Loss: 1.951299
[LBFGS] Step 70/100 | Loss: 1.853190
[LBFGS] Step 75/100 | Loss: 1.728588
[LBFGS] Step 80/100 | Loss: 1.622905
[LBFGS] Step 85/100 | Loss: 1.553065
[LBFGS] Step 90/100 | Loss: 1.493478
[LBFGS] Step 95/100 | Loss: 1.439807
Model saved as 'pinn_model.pth'.
Test MSE: 0.08740921318531036
"""

# =============================================================================
# 1. PARSE THE LOG
# =============================================================================

adam_pattern = re.compile(
    r"\[Adam\] Epoch (\d+)/\d+ \| Total Loss: ([\d.]+), PDE: ([\d.]+), BC_top: ([\d.]+), BC_bot: ([\d.]+), IC: ([\d.]+), Data: ([\d.]+)"
)
lbfgs_pattern = re.compile(r"\[LBFGS\] Step (\d+)/\d+ \| Loss: ([\d.]+)")

# Data containers for Adam
adam_epochs = []
adam_total = []
adam_pde = []
adam_bc_top = []
adam_bc_bot = []
adam_ic = []
adam_data = []

# Data containers for LBFGS
lbfgs_steps = []
lbfgs_loss = []

for line in raw_log.splitlines():
    # Match Adam log
    match_adam = adam_pattern.search(line)
    if match_adam:
        epoch = int(match_adam.group(1))
        total_loss = float(match_adam.group(2))
        pde_loss = float(match_adam.group(3))
        bc_top_loss = float(match_adam.group(4))
        bc_bot_loss = float(match_adam.group(5))
        ic_loss = float(match_adam.group(6))
        data_loss = float(match_adam.group(7))

        adam_epochs.append(epoch)
        adam_total.append(total_loss)
        adam_pde.append(pde_loss)
        adam_bc_top.append(bc_top_loss)
        adam_bc_bot.append(bc_bot_loss)
        adam_ic.append(ic_loss)
        adam_data.append(data_loss)

    # Match LBFGS log
    match_lbfgs = lbfgs_pattern.search(line)
    if match_lbfgs:
        step = int(match_lbfgs.group(1))
        loss = float(match_lbfgs.group(2))

        lbfgs_steps.append(step)
        lbfgs_loss.append(loss)

# =============================================================================
# 2. PLOT THE RESULTS
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- Left subplot: Adam losses ---
ax1.plot(adam_epochs, adam_total, label='Total Loss', color='black', linewidth=2)
ax1.plot(adam_epochs, adam_pde, label='PDE Loss', linestyle='--')
ax1.plot(adam_epochs, adam_bc_top, label='BC_top Loss', linestyle='--')
ax1.plot(adam_epochs, adam_bc_bot, label='BC_bot Loss', linestyle='--')
ax1.plot(adam_epochs, adam_ic, label='IC Loss', linestyle='--')
ax1.plot(adam_epochs, adam_data, label='Data Loss', linestyle='--')

ax1.set_title('Adam Training')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')  # Log scale for losses (optional)
ax1.legend(loc='upper right')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# --- Right subplot: LBFGS total loss ---
ax2.plot(lbfgs_steps, lbfgs_loss, color='blue', label='LBFGS Total Loss')
ax2.set_title('LBFGS Training')
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')  # Log scale for losses (optional)
ax2.legend(loc='upper right')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
