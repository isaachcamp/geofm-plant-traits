
from typing import Tuple
from numpy import ndarray
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel


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

class RFBandsOnly(BaseModel):
    name = "RF – bands only"

    def __init__(self, seed, var):
        super().__init__(seed)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            max_features=0.5,
            min_samples_leaf=1,
            random_state=seed
        )

    def fit(self, X: Array, y: Array, X_val: Array, y_val: Array) -> None:
        """Fit the model to the data."""
        X = np.concatenate([X, X_val])
        y = np.concatenate([y, y_val])
        return self.model.fit(X, y)

    def predict(self, X: Array) -> Array:
        """Make predictions using the model."""
        return self.model.predict(X)

    @staticmethod
    def configure_data(X: pd.DataFrame, y: Array) -> Tuple[Array, Array]:
        """Configure the data for the model."""
        return X[BANDS].to_numpy(), y.to_numpy().ravel()


def create_model(seed=None, var=None) -> RFBandsOnly:
    """Create and return a model instance."""
    return RFBandsOnly(seed, var)
