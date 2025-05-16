

import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_size=10, latent_dim=4):
        super(VAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, input_size),
        )

        self.mean_layer = nn.Linear(40, latent_dim)
        self.var_layer = nn.Linear (40, latent_dim)

    def encode(self, x):
        encoded = self.encoder(x)
        mean = self.mean_layer(encoded)
        var = torch.exp(0.5 * self.var_layer(encoded))
        z = self.reparameterization(mean, var)
        return z

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decoder(z)
        return decoded, z
