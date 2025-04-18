
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils import standardise


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

class NNBandsOnly(BaseEstimator, RegressorMixin):
    def __init__(self, seed=None):
        self.name = "Vanilla NN using only spectral bands"

        self.seed = seed
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y):
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        # Train for a fixed number of epochs
        self.model.train()
        loss_history = []

        for epoch in range(100):
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

            # Average loss over the epoch
            loss = running_loss / len(dataloader)
            loss_history.append(loss)

            # Print loss every 5 epochs
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return self

    def predict(self, X):
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)

        return preds.cpu().numpy()

    def configure_data(self, X, y):
        """Configure the data for the model."""
        # Convert to PyTorch tensors
        X = standardise(X[BANDS].to_numpy(), axis=1)
        y = standardise(y.to_numpy().reshape(-1, 1))
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        return X, y

def create_model():
    return NNBandsOnly()
