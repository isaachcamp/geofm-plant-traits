
from typing import Dict
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
import torch


# define types
Tensor = torch.Tensor
Array = ndarray


class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        self.stats = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
            self,
            X: Tensor | Array,
            y: Tensor | Array,
            X_val: Tensor | Array,
            y_val: Tensor | Array
    ) -> None:
        """Model training loop."""

    def predict(self, X: Tensor | Array):
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

    def validation(self, inputs: Tensor, targets: Tensor):
        """Validate model not over-fitting."""
