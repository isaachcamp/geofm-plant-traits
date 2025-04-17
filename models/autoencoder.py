

from torch import nn


class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_size=10, latent_dim=2, in_channels=1):
        super(ConvAutoencoder1D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),

            # Second convolutional layer
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            # Third convolutional layer
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            # Fourth convolutional layer
            nn.Conv1d(in_channels=64, out_channels=latent_dim, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2)
        )

        # Size of the encoder output
        self.encoded_size = input_size // 2  # 1 layer with stride 2

        # Decoder
        self.decoder = nn.Sequential(
            # First transposed convolutional layer
            nn.ConvTranspose1d(
                in_channels=latent_dim, out_channels=64, kernel_size=3,
                stride=1, padding=0, output_padding=0
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            # Second transposed convolutional layer
            nn.ConvTranspose1d(
                in_channels=64, out_channels=32, kernel_size=3,
                stride=1, padding=1, output_padding=0
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            # Third transposed convolutional layer
            nn.ConvTranspose1d(
                in_channels=32, out_channels=16, kernel_size=3,
                stride=1, padding=1, output_padding=0
            ),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            # Fourth transposed convolutional layer
            nn.ConvTranspose1d(
                in_channels=16, out_channels=in_channels, kernel_size=5,
                stride=2, padding=2, output_padding=1
            ),
            # nn.Tanh()  # Output activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.shape)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x):
        """Encode the input data into the latent space."""
        return self.encoder(x)
