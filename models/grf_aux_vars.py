
from pathlib import Path
import json

from typing import Tuple
from numpy import ndarray
import pandas as pd
from PyGRF import PyGRF

from models.base_model import BaseModel


Array = ndarray


HPARAMS_PATH = Path('./data/metadata/grf_hparams.json')
COORDS_PATH = Path('./data/metadata/pixel_coords.csv')

class GRFAuxAndBands(BaseModel):
    name = "GRF – bands and aux vars"
    coords: pd.DataFrame

    def __init__(self, seed, var):
        super().__init__(seed)

        hparams = self.get_hparams(var)
        self.bandwidth = hparams['bandwidth']
        self.local_weight = hparams['local_weight']

        self.model = PyGRF.PyGRFBuilder(
            n_estimators=100,
            max_features=0.5,
            band_width=self.bandwidth,
            train_weighted=True,
            predict_weighted=True,
            bootstrap=True,
            resampled=False,
            random_state=seed
        )

    def fit(self, X: Array, y: Array, X_val: Array, y_val: Array) -> None:
        """Fit the model to the data."""
        coords = self.get_coords(y)
        return self.model.fit(X, y, coords[['Lon', 'Lat']])

    def predict(self, X: Array) -> Array:
        """Make predictions using the model."""
        coords = self.get_coords(X)
        predict_combined, predict_global, predict_local = self.model.predict(
            X,
            coords[['Lon', 'Lat']],
            local_weight=self.local_weight
        )
        predict_combined = pd.Series(predict_combined, index=X.index)
        return predict_combined

    @staticmethod
    def configure_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Configure the data for the model."""
        return X, y

    def get_hparams(self, var) -> dict:
        """Get the hyperparameters of the model."""
        with open(HPARAMS_PATH, 'r') as f:
            hparams = json.load(f)
        return hparams[var]

    def get_coords(self, y: pd.DataFrame) -> None:
        """Get the coordinates of the model."""
        all_coords = pd.read_csv(COORDS_PATH, index_col=0)
        return all_coords.loc[y.index]


def create_model(seed=None, var=None) -> GRFAuxAndBands:
    """Create and return a model instance."""
    return GRFAuxAndBands(seed, var)
