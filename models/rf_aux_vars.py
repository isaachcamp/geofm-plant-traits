
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel


Array = ndarray

class RFAuxAndBands(BaseModel):
    """Random Forest model that uses auxiliary variables and spectral bands."""
    name = "Random Forest using spectral bands and auxiliary variables"

    def __init__(self, seed, var):
        super().__init__(seed)
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            max_features="log2",
            min_samples_leaf=2,
            random_state=seed
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


def create_model(seed=None, var=None) -> RFAuxAndBands:
    """Create and return a model instance."""
    return RFAuxAndBands(seed, var)
