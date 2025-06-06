
import random
from pathlib import Path
import numpy as np
import pandas as pd


class LabelledTraitData:
    """Class to handle trait data."""
    def __init__(self, data_path: Path, var: str):
        self.data_path = data_path
        self.var = var
        self.load_data()
        self.standardise()

    def load_data(self):
        """Load the trait data from a CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory {self.data_path} does not exist.")

        # Load the datasets and labels.
        self.train_data = pd.read_csv(
            self.data_path / 'train' / f'{self.var}_train_data.csv',
            index_col=0
        )
        self.train_labels = pd.read_csv(
            self.data_path / 'train' / f'{self.var}_train_labels.csv',
            index_col=0
        )

        self.val_data = pd.read_csv(
            self.data_path / 'validation' / f'{self.var}_val_data.csv',
            index_col=0
        )
        self.val_labels = pd.read_csv(
            self.data_path / 'validation' / f'{self.var}_val_labels.csv',
            index_col=0
        )

        self.test_data = pd.read_csv(
            self.data_path / 'test' / f'{self.var}_test_data.csv',
            index_col=0
        )
        self.test_labels = pd.read_csv(
            self.data_path / 'test' / f'{self.var}_test_labels.csv',
            index_col=0
        )

    @staticmethod
    def _standardise(train, val, test):
        train_idx, val_idx, test_idx = train.index, val.index, test.index

        df = pd.concat([train, val, test])
        std_df = (df - df.mean(axis=0)) / df.std(axis=0)

        return std_df.loc[train_idx], std_df.loc[val_idx], std_df.loc[test_idx]

    def standardise(self):
        """Standardise both predictors and targets."""
        self.train_data, self.val_data, self.test_data = self._standardise(
            self.train_data, self.val_data, self.test_data
        )
        self.train_labels, self.val_labels, self.test_labels = self._standardise(
            self.train_labels, self.val_labels, self.test_labels
        )


def train_val_test_split(df: pd.DataFrame, train_split: float = 0.7, validation_split: float = 0.1):
    """Split dataset into training, validation, and test sets."""
    data_size = df.shape[0]

    train_size = np.ceil(train_split * data_size).astype(int)
    validation_size = np.ceil(validation_split * data_size).astype(int)

    # Select random pixels
    train_pixels = random.sample(list(df.index), train_size)
    validation_pixels = random.sample(list(set(df.index) - set(train_pixels)), validation_size)
    test_pixels = list(set(df.index) - set(train_pixels) - set(validation_pixels))

    return df.loc[train_pixels, :], df.loc[validation_pixels, :], df.loc[test_pixels, :]

def get_outlier_zscore(X: pd.Series, zlim: int = 5) -> pd.DataFrame:
    """Get outliers from the dataset using provided Z-score as threshold."""
    thresh = zlim * X.std()
    return X[( X > (X.mean() + thresh) ) | ( X < (X.mean() - thresh) )]

def get_outlier_iqr(X: pd.Series, zlim: int = 1.5) -> pd.DataFrame:
    """Get outliers outside upper and lower fences."""
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    thresh = zlim * (q3 - q1)
    return X[( X > (q3 + thresh) ) | ( X < (q1 - thresh) )]

def minmax_scaler(X: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Normalize the dataset using min-max scaling."""
    if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
        raise TypeError("Input must be a pandas DataFrame or Series.")
    return (X - X.min()) / (X.max() - X.min())

def standardise(X: np.ndarray, axis=0) -> np.ndarray:
    """Standardize the dataset using z-score normalization."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    return (X - X.mean(axis=axis)) / X.std(axis=axis)
