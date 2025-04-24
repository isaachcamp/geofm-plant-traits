

from sklearn.ensemble import RandomForestRegressor

class RFAuxAndBands:
    """Random Forest model that uses auxiliary variables and spectral bands."""
    name = "Random Forest using spectral bands and auxiliary variables"

    def __init__(self, seed):
        self.model = RandomForestRegressor(n_estimators=100, random_state=seed)

    def fit(self, X, y, X_val, y_val):
        """Fit the model to the data."""
        return self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X)

    def configure_data(self, X, y):
        """Configure the data for the model."""
        return X, y.to_numpy().ravel()


def create_model(seed=None):
    """Create and return a model instance."""
    return RFAuxAndBands(seed)
