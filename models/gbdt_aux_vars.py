
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor

from models.base_model import BaseModel


Array = ndarray


class GBDTAuxAndBands(BaseModel):
    name = "Gradient boosted decision trees using spectral bands and auxiliary variables"

    def __init__(self, seed, var):
        super().__init__(seed)
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=seed,
            max_features="sqrt",
            subsample=0.8,
        )

    def fit(self, X: Array, y: Array, X_val: Array, y_val: Array) -> None:
        """Fit the model to the data."""
        return self.model.fit(X, y)

    def predict(self, X: Array) -> Array:
        """Make predictions using the model."""
        return self.model.predict(X)

    @staticmethod
    def configure_data(X: DataFrame, y: Array) -> Tuple[Array, Array]:
        """Configure the data for the model."""
        return X.to_numpy(), y.to_numpy().ravel()


def create_model(seed=None, var=None) -> GBDTAuxAndBands:
    """Create and return a model instance."""
    return GBDTAuxAndBands(seed, var)
