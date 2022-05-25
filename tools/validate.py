import pandas as pd
import numpy as np
import torch
from scipy import stats


feature_cols = ["f_{}".format(i) for i in range(0, 300)]
data = pd.read_pickle('/home/ubuntu/data/supplemental_train.parquet')
data_times = data.time_id.unique()
data['predicted'] = 0

def validate(model):
    model.eval()
    for data_time in data_times:
        curr_data = data[data.time_id == data_time]

        features = torch.tensor(curr_data[feature_cols].values).float()
        investment_ids = torch.tensor(curr_data['investment_id'].astype(int).values)
        time_ids = torch.tensor(curr_data['time_id'].astype(int).values).unsqueeze(1)

        batch = {
            'features': features,
            'investment_ids': investment_ids,
            'time_ids': time_ids
        }

        results = model.predict_step(batch)
        data.loc[data.time_id == data_time, 'predicted'] = results.detach()
    return data[['time_id', 'target', 'predicted']].groupby("time_id").apply(lambda item: stats.pearsonr(item.target, item.predicted)[0] if len(item.target)>1 else 1).mean()