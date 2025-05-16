
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import mean_absolute_percentage_error
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.base_model import BaseModel

# define types
Tensor = torch.Tensor
Array = ndarray

BANDS = [
    'B2_real', # Blue band, 490 nm
    'B3_real', # Green band, 560 nm
    'B4_real', # Red band, 665 nm
    'B5_real', # Red edge band, 705 nm
    'B6_real', # Red edge band, 740 nm
    'B7_real', # Red edge band, 783 nm
    'B8_real', # NIR band, 842 nm
    'B11_real', # SWIR band, 1610 nm
    'B12_real', # SWIR band, 2190 nm
    # 'B8a_real' # NIR band, 865 nm
]

class Autoencoder(nn.Module):
    """Simple autoencoder model, trained in separate notebook."""
    def __init__(self, input_size=10, latent_dim=4):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.shape)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AEBandsOnly(BaseModel):
    def __init__(self, seed, var):
        self.name = "AE + LR head – bands only"
        self.latent_dim = 6

        super().__init__(seed)

        self.model = self.model_setup()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.05)
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)

    def model_setup(self):
        model = Autoencoder(input_size=9, latent_dim=self.latent_dim)
        model.load_state_dict(torch.load('./weights/ae_model.pth', weights_only=True))
        model.eval()

        # Swap decoder for linear regression head.
        model.decoder = nn.Linear(self.latent_dim, 1)
        model.to(self.device)

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        return model

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        # Create DataLoaders
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

        # Train for a fixed number of epochs
        self.model.train()
        loss_history = []

        for epoch in range(100):
            running_loss = 0.
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero gradients, forward pass, backward pass, optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)[0]
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Validation
            val_loss, val_mape = self.validation(X_val, y_val)
            loss_history.append(val_loss)

            # Print loss every 5 epochs
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {val_loss}, MAPE: {val_mape}")

        return self

    def predict(self, X: Tensor) -> Tensor:
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)[0]
        return preds.cpu()

    def configure_data(self, X: DataFrame, y: DataFrame) -> Tuple[Tensor]:
        """Configure the data for the model."""
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X[BANDS].to_numpy())
        y = torch.FloatTensor(y.to_numpy().reshape(-1, 1))
        return X, y

    def validation(self, inputs: Tensor, targets: Tensor) -> Tuple[float]:
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            targets = targets.float().to(self.device)
            preds = self.model(inputs)[0]

        val_loss = self.loss_fn(preds, targets).item()

        # Unstandardise the predictions and targets for MAPE calculation
        preds, targets = self.unstandardise(preds, targets)
        val_mape = mean_absolute_percentage_error(preds, targets)

        return val_loss, val_mape

def create_model(seed=None, var=None) -> AEBandsOnly:
    return AEBandsOnly(seed, var)
