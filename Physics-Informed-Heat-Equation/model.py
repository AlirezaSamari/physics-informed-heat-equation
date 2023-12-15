import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, n_layer):
        super().__init__()
        self.activation = nn.Tanh()
        # Input
        self.fc_i = nn.Linear(n_in, n_hidden)

        # Hidden Layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layer - 1):
            hidden_layer = nn.Linear(n_hidden, n_hidden)
            self.hidden_layers.extend([hidden_layer, self.activation])

        # Output
        self.fc_o = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.fc_i(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc_o(x)
        return x
