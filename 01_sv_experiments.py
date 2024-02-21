from config import *

import numpy as np
import os
from smote_variants.base import OverSampling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from metrics import BinaryConfusionMatrix
from sklearn.compose import ColumnTransformer

from tqdm import tqdm
import pandas as pd

METRICS = ["recall", "precision"]

for ds_name in DATASETS:
    if f"{ds_name}.npy" in os.listdir('results'):
        print(f"{ds_name} done")
        continue

    scores = np.zeros(
        shape=(len(SAMPLING), len(CLASSIFIERS), RSKF.get_n_splits(), len(METRICS))
    )

    X, y = DATASETS[ds_name]["data"], DATASETS[ds_name]["target"]

    categorical_features = X.select_dtypes(include=['category']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    ohe = ColumnTransformer(
        transformers=[("ohe", encoder, categorical_features),],
        remainder="passthrough",
    )

    X = ohe.fit_transform(X)

    X.astype(float)
    y = LabelEncoder().fit_transform(y)

    bar = tqdm(desc=ds_name, total=RSKF.get_n_splits(), leave=True)

    for s_idx, (train, test) in enumerate(RSKF.split(X, y)):
        scaler = StandardScaler()
        X_scale = scaler.fit_transform(X[train])

        bar_sampling = tqdm(SAMPLING, leave=False)

        for o_idx, _ovs in enumerate(bar_sampling):
            bar_sampling.set_description(_ovs.__name__)
            ovs = _ovs()
            X_train, y_train = ovs.fit_resample(X_scale, y[train])

            for c_idx, _clf in enumerate(CLASSIFIERS):
                clf = _clf()

                if len(X_train) < 5:
                    clf = _clf(n_neighbors=len(X_train))
                    print(_ovs.__name__, X_train.shape)

                clf.fit(X_train, y_train)

                X_test = (
                    ovs.preprocessing_transform(scaler.transform(X[test]))
                    if OverSampling.cat_dim_reduction in ovs.categories
                    else scaler.transform(X[test])
                )

                y_pred = clf.predict(X_test)

                scores[o_idx, c_idx, s_idx] = BinaryConfusionMatrix(
                    y[test], y_pred
                ).get_metrics(METRICS)

        bar.update(1)

    bar.close()

    np.save(f"results/{ds_name}", scores)
