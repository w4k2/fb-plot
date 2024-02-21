import numpy as np
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from config import *
from operator import itemgetter

table = []

for ds in DATASETS:
    X, y = DATASETS[ds]['data'], DATASETS[ds]['target']
    y = LabelEncoder().fit_transform(y)

    n_samples = len(X)
    n_features = len(X.T)
    c_lab, c_num = np.unique(y, return_counts=True)
    ir = np.max(c_num) / np.min(c_num)
    table.append([ds, n_samples, n_features, ir, np.min(c_num)])

table = list(sorted(table, key=itemgetter(3)))

with open('datasets.info', 'w') as fp:
    fp.write(tabulate(table, tablefmt='grid'))
    fp.write(f"Total: {len(DATASETS)}")

print(tabulate(table, tablefmt='latex'))
