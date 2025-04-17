
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


class SentinelSpectraDataset(Dataset):
    def __init__(self, dpath: str):
        self.dpath = Path(dpath)
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.dpath).drop(columns='chipid')
        return self.transform(data)

    def transform(self, data):
        data = torch.tensor(
            data.to_numpy().reshape(data.shape[0], 1, 10),
            dtype=torch.float32
        )
        return normalize(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return [self.data[idx]]

def min_max_norm(data):
    """
    Normalize the data to the range [0, 1].
    """
    dmin, dmax = 0, 10000
    return (data - dmin) / (dmax - dmin)

def normalize(data):
    """
    Normalize the data to the range [0, 1].
    """
    return (data - data.mean(axis=0)) / data.std(axis=0)
