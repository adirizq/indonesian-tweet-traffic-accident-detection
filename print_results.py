import os
import sys
import pandas as pd

from tqdm import tqdm

root = 'csv_logs'
architectures = os.listdir(root)

data = {
    'model': [],
    'hyper_params': [],
    'accuracy': [],
    'f1': [],
    'precision': [],
    'recall': [],
}

for architecture in tqdm(architectures):
    hyper_params = os.listdir(os.path.join(root, architecture))
    for hyper_param in hyper_params:
        all_metrics = pd.read_csv(f'csv_logs/{architecture}/{hyper_param}/version_0/metrics.csv')
        test = all_metrics.dropna(subset=['test_loss']).dropna(axis='columns').drop(['step', 'epoch'], axis=1)
        data['model'].append(architecture)
        data['hyper_params'].append(hyper_param)
        data['accuracy'].append(round(test['test_accuracy'].values[0] * 100, 2))
        data['f1'].append(round(test['test_f1_score'].values[0] * 100, 2))
        data['precision'].append(round(test['test_precision'].values[0] * 100, 2))
        data['recall'].append(round(test['test_recall'].values[0] * 100, 2))

df = pd.DataFrame(data)
df.to_csv('results.csv', index=False)
