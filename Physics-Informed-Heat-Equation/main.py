import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import PINN
from dataset import generate_heat_equation_dataset
from losses import pde_loss
from trainer import pinn_trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
alpha = 0.001
nx, ny = 40, 40
nt = 40

# Generate dataset
X, X_train, T_train = generate_heat_equation_dataset(nx, ny, nt)
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of T_train:", T_train.shape)

# Create PINN model
model = PINN(3, 64, 1, 6).to(device)

# Define optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train the PINN model
pinn_trainer(10000, model, optimizer, criterion, X, X_train, T_train, device)

# Visualization
with torch.no_grad():
    y_pred = model(X.to(device))
    y_pred = y_pred.reshape(nx, ny, nt).cpu().numpy()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
time_steps = [1, nt // 4, nt // 2, nt - 1]

for i, ax in zip(time_steps, axs.flatten()):
    ax.contourf(y_pred[:, :, i].T, cmap='jet')
    ax.set_title(f'Temp Field at t={i+1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

plt.colorbar(axs[0, 0].contourf(y_pred[:, :, 1].T, cmap='jet'), ax=axs, orientation='horizontal', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
