import torch

def generate_heat_equation_dataset(nx, ny, nt):
    # Geo
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    t = torch.linspace(0, 10, nt)
    X = torch.stack(torch.meshgrid(x, y, t)).reshape(3, -1).T.requires_grad_(True)

    # IC & BC
    X_BC_0 = X[(X[:, 0] == x[0]) | (X[:, 0] == x[-1]) | (X[:, 1] == y[0])]
    X_BC_1 = X[(X[:, 1] == y[-1]) & (X[:, 2] > t[0])]
    X_IC = X[X[:, 2] == t[0]]
    X_train = torch.cat([X_BC_0, X_BC_1, X_IC])
    T_BC_0 = torch.zeros(len(X_BC_0)).view(-1, 1)
    T_BC_1 = torch.sin(torch.pi * X_BC_1[:, 0]).view(-1, 1)
    T_IC = torch.zeros([len(X_IC)]).view(-1, 1)
    T_train = torch.cat([T_BC_0, T_BC_1, T_IC])

    return X, X_train, T_train