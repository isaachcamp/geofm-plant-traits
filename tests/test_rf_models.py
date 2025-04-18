
import numpy as np
import pandas as pd
from models.rf_bands_only import RFBandsOnly
from models.rf_aux_vars import RFAuxAndBands

# Test data for the models
X = pd.DataFrame({
    'B2_real': [1, 2, 3],
    'B3_real': [4, 5, 6],
    'B4_real': [7, 8, 9],
    'B5_real': [10, 11, 12],
    'B6_real': [13, 14, 15],
    'B7_real': [16, 17, 18],
    'B8_real': [19, 20, 21],
    'B11_real': [22, 23, 24],
    'B12_real': [25, 26, 27],
    'aux1': [28, 29, 30],
    'aux2': [31, 32, 33],
})
y = pd.Series([1, 2, 3])

# Feed the test data to the models
# and check if they can handle it without errors.
# This is a simple test to ensure that the models can be instantiated and
# configured correctly with the provided data.
# The actual model fitting and prediction are not tested here, as they would
# require a more complex setup with training and test datasets.

# Test the RFBandsOnly model
class TestRFBandsOnly:
    def test_init(self):
        """Test the initialization of the RFBandsOnly model."""
        model = RFBandsOnly(seed=42)
        assert model.name == "Random Forest using only spectral bands"
        assert model.model is not None
        assert model.model.n_estimators == 100
        assert model.model.random_state == 42

    def test_configure_data(self):
        """Test the configure_data method of the RFBandsOnly model."""
        model = RFBandsOnly(seed=42)
        X_configured, y_configured = model.configure_data(X, y)

        assert X_configured.shape[1] == 9  # Number of bands
        assert y_configured.shape == (3,)

        assert isinstance(X_configured, pd.DataFrame)
        assert isinstance(y_configured, np.ndarray)

class TestRFAuxAndBands:
    def test_init(self):
        """Test the initialization of the RFAuxAndBands model."""
        model = RFAuxAndBands(seed=42)
        assert model.name == "Random Forest using spectral bands and auxiliary variables"
        assert model.model is not None
        assert model.model.n_estimators == 100
        assert model.model.random_state == 42

    def test_configure_data(self):
        """Test the configure_data method of the RFAuxAndBands model."""
        model = RFAuxAndBands(seed=42)
        X_configured, y_configured = model.configure_data(X, y)

        assert X_configured.shape[1] == 11  # Number of bands + auxiliary variables
        assert y_configured.shape == (3,)

        assert isinstance(X_configured, pd.DataFrame)
        assert isinstance(y_configured, np.ndarray)
