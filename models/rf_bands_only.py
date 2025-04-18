

from sklearn.ensemble import RandomForestRegressor
from src.data_utils import get_outlier_iqr

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

class RFBandsOnly:
    name = "Random Forest using only spectral bands"

    def __init__(self, seed):
        self.model = RandomForestRegressor(n_estimators=100, random_state=seed)

    def fit(self, X, y):
        """Fit the model to the data."""
        return self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)

    def configure_data(self, X, y):
        """Configure the data for the model."""
        return X[BANDS], y.to_numpy().ravel()


def create_model(seed=None):
    """Create and return a model instance."""
    return RFBandsOnly(seed)
