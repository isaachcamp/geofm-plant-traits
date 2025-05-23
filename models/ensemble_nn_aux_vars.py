
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame
import torch

from models.base_model import BaseModel
from models.nn_aux_vars import NNAuxAndBands


# define types
Tensor = torch.Tensor
Array = ndarray


class NNEnsembleAuxAndBands(BaseModel):
    name = "Ensemble NN – bands and aux vars"

    def __init__(self, seed, var):
        # Ignore seed else weak estimators will be the same.
        self.var = var

        # Create an ensemble of models, initializing each with different weights.
        self.total_models = 50
        self.models = [NNAuxAndBands(seed, var) for _ in range(self.total_models)]
        for model in self.models:
            model.total_epochs = 10

        super().__init__(seed)

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        for i, model in enumerate(self.models):
            print(f"Model {i} - Training...")
            model.set_stats(self.stats)
            model.fit(X, y, X_val, y_val)

        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        Make predictions using the ensemble of snapshots.
        """
        inputs = X.to(self.device)
        outputs = []
        for model in self.models:
            outputs.append(model.predict(inputs))

        return torch.mean(torch.stack(outputs), dim=0).cpu()

    def configure_data(self, X: DataFrame, y: DataFrame) -> Tuple[Tensor]:
        """Configure the data for the model."""
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X.to_numpy())
        y = torch.FloatTensor(y.to_numpy().reshape(-1, 1))
        return X, y


def create_model(seed=None, var=None):
    return NNEnsembleAuxAndBands(seed, var)
