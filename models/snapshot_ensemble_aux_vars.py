
import math
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import mean_absolute_percentage_error
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from models.base_model import BaseModel


# define types
Tensor = torch.Tensor
Array = ndarray


class NNSnapshotEnsembleAuxAndBands(BaseModel):
    def __init__(self, seed, var):
        self.name = "Snapshot Ensemble NN – bands and aux vars"

        self.total_epochs = 3000
        self.total_snapshots = 30
        self.lr_max = 1e-3
        self.snapshots = []

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.05)
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)

    def cosine_annealing_lr(self, epoch):
        """
        Compute the learning rate using cosine annealing.
        """
        length_scale = self.total_epochs // self.total_snapshots
        return self.lr_max * 0.5 * (1 + math.cos(math.pi * (epoch % length_scale) / length_scale))

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        # Create DataLoaders
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

        # Train for a fixed number of epochs
        self.model.train()
        loss_history = []

        for epoch in range(self.total_epochs):
            running_loss = 0.

            lr = self.cosine_annealing_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

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

            # Save snapshot at the end of each cycle
            if (epoch + 1) % (self.total_epochs // self.total_snapshots) == 0:
                self.snapshots.append(self.model.state_dict())

        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        Make predictions using the ensemble of snapshots.
        """
        self.model.eval()
        inputs = X.to(self.device)
        outputs = []
        with torch.no_grad():
            for snapshot in self.snapshots:
                self.model.load_state_dict(snapshot)
                outputs.append(self.model(inputs))
        return torch.mean(torch.stack(outputs), dim=0).cpu()

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

        val_loss = self.loss_fn(preds, targets).item()

        # Unstandardise the predictions and targets for MAPE calculation
        preds, targets = self.unstandardise(preds, targets)
        val_mape = mean_absolute_percentage_error(preds, targets)

        return val_loss, val_mape


def create_model(seed=None, var=None):
    return NNSnapshotEnsembleAuxAndBands(seed, var)
