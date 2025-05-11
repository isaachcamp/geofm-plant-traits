
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
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
    'B12_real' # SWIR band, 2190 nm
    # 'B8a_real' # NIR band, 865 nm
]

class CNNBandsOnly(BaseModel):
    def __init__(self, seed, var):
        self.name = "CNN using only spectral bands"

        super().__init__(seed)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1), # (n, 8, 9)
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1), # (n, 16, 9)
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (n, 32, 9)
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Flatten(),
            nn.Linear(288, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.05)
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        # Create DataLoaders
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

        # Train for a fixed number of epochs
        self.model.train()
        loss_history = []

        for epoch in range(300):
            running_loss = 0.
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Zero gradients, forward pass, backward pass, optimize
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
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
            preds = self.model(X)

        return preds.cpu()

    def configure_data(self, X: DataFrame, y: DataFrame) -> Tuple[Tensor]:
        """Configure the data for the model."""
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X[BANDS].to_numpy())
        X = X.reshape(X.shape[0], 1, X.shape[1]) # add channel dimension
        y = torch.FloatTensor(y.to_numpy().reshape(-1, 1))
        return X, y

    def validation(self, inputs: Tensor, targets: Tensor) -> Tuple[float]:
        self.model.eval()

        with torch.no_grad():
            inputs = inputs.to(self.device)
            targets = targets.float().to(self.device)
            preds = self.model(inputs)

            preds, targets = self.unstandardise(preds, targets)

        val_loss = self.loss_fn(preds, targets).item()
        val_mape = mean_absolute_percentage_error(preds, targets)

        return val_loss, val_mape

def create_model(seed=None, var=None) -> CNNBandsOnly:
    return CNNBandsOnly(seed, var)
