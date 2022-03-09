import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SimpleMarketDataset(Dataset):
    def __init__(self, pkl_file):
        super().__init__()
        self.data = pd.read_pickle(pkl_file)
        cols = [col for col in self.data.columns if not col.find('Unnamed')]
        self.data = self.data.drop(cols, axis=1)
        self.target = self.data['target'].values
        self.features = self.data.drop(['row_id', 'target', 'investment_id'], 1).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
         return: torch.tensor([features, target])
        """
        return torch.tensor(self.features[index]).float(), torch.tensor(self.target[index]).float()
