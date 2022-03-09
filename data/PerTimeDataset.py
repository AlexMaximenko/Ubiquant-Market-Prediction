import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PerTimeDataset(Dataset):
    """
    Returns a sample containing all the data for a next time_id instead of single samples.
    """
    def __init__(self, pkl_file_path):
        super().__init__()
        self.data = pd.read_pickle(pkl_file_path)
        cols = [col for col in self.data.columns if not col.find('Unnamed')]
        self.data = self.data.drop(cols, axis=1)
        self.time_ids = np.sort(self.data['time_id'].unique())
        self.feature_cols = ["f_{}".format(i) for i in range(0, 300)]


    def __len__(self):
        return len(self.time_ids)

    def __getitem__(self, index):
        """
        Return all data with time_id == self.time_ids[index]
        """
        temp_data = self.data[self.data['time_id'] == self.time_ids[index]]
        features = torch.tensor(temp_data[self.feature_cols].values).float()
        target = torch.tensor(temp_data['target'].values).float()
        time_ids = torch.tensor(temp_data['time_id'].values)
        investment_ids = torch.tensor(temp_data['investment_id'].values)

        return {
            'features': features, 
            'target': target,
            'investment_id': investment_ids,
            'time_id': time_ids
        }