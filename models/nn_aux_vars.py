
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


class NNAuxAndBands(BaseModel):
    def __init__(self, seed=None):
        self.name = "NN using spectral bands and auxiliary variables"

        super().__init__(seed)

        self.model = nn.Sequential(
            nn.Linear(26, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        # Create DataLoaders
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

        # Train for a fixed number of epochs
        self.model.train()
        loss_history = []

        for epoch in range(300):
            running_loss = 0.
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero gradients, forward pass, backward pass, optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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
            preds = self.model(X)
        return preds.cpu()

    def configure_data(self, X: DataFrame, y: DataFrame) -> Tuple[Tensor]:
        """Configure the data for the model."""
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X.to_numpy())
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

def create_model(seed=None):
    return NNAuxAndBands(seed)
