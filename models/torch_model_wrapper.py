
from typing import Dict
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
import torch
from torch.utils.data import DataLoader

# define types
Tensor = torch.Tensor
Array = ndarray


class TorchModel(BaseEstimator, RegressorMixin):
    def __init__(self, seed=None):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = None

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        """Model training loop."""

    def predict(self, X: Tensor):
        """Model forward pass to make predictions."""

    def set_stats(self, stats: Dict[str, float]):
        """Update stats (mean and std) for trait data."""
        self.stats = stats

    @staticmethod
    def _unstandardise(x: Array | Tensor, mean: float, std: float):
        return (x * std) + mean

    def unstandardise(self, preds: Tensor, targets: Tensor):
        """Rescale targets and predictions for sensible metrics."""
        preds = self._unstandardise(preds, self.stats['mean'], self.stats['std'])
        targets = self._unstandardise(targets, self.stats['mean'], self.stats['std'])
        return preds, targets

    def validation(self, val_dataloader: DataLoader):
        """Validate model not over-fitting."""
