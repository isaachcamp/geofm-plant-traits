
import torch
import torch.nn as nn
import pandas as pd
from models.vanilla_nn_bands_only import NNBandsOnly

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
})
y = pd.Series([1, 2, 3])

class TestNNBandsOnly:
    def test_init(self):
        model = NNBandsOnly(seed=42)

        assert model.seed == 42
        assert isinstance(model.model, nn.Sequential)
        assert isinstance(model.optimizer, torch.optim.Adam)
        assert isinstance(model.loss_fn, nn.MSELoss)
        assert model.device.type in ["cuda", "cpu"]

    def test_predict_handles_batch_correctly(self):
        model = NNBandsOnly(seed=42)

        X_train, y_train = model.configure_data(X, y)
        model.fit(X_train, y_train)

        preds = model.predict(X_train)

        # Check if predictions are made
        assert preds.shape == torch.Size([3, 1])

    def test_configure_data(self):
        """Test the configure_data method of the NNBandsOnly model."""
        model = NNBandsOnly(seed=42)
        X_configured, y_configured = model.configure_data(X, y)

        # Check if the data is configured correctly
        assert isinstance(X_configured, torch.Tensor)
        assert isinstance(y_configured, torch.Tensor)

        assert X_configured.shape == (3, 9)
        assert y_configured.shape == torch.Size([3, 1])

        # Check if the data is scaled
        assert torch.all(X_configured >= 0) and torch.all(X_configured <= 1)
