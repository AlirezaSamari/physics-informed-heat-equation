import torch

def pde_loss(X, T_hat, alpha, criterion):
    # PDE loss
    dT_dX = torch.autograd.grad(T_hat, X, torch.ones_like(T_hat), create_graph=True)[0]
    dT_dt = dT_dX[:, 2]
    
    # Laplacian terms
    dT_dxx = torch.autograd.grad(dT_dX[:, 0], X, torch.ones_like(dT_dX[:, 0]), create_graph=True)[0][:, 0]
    dT_dyy = torch.autograd.grad(dT_dX[:, 1], X, torch.ones_like(dT_dX[:, 1]), create_graph=True)[0][:, 1]
    
    return criterion(alpha * (dT_dxx + dT_dyy), dT_dt)
