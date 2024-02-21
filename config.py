import os

from smote_variants.oversampling import *

# use
SAMPLING = [
    ADASYN,
    ADG,
    ADOMS,
    AHC,
    AMSCO,
    AND_SMOTE,
    ANS,
    ASMOBD,
    A_SUWO,
    Assembled_SMOTE,
    Borderline_SMOTE1,
    Borderline_SMOTE2,
    CBSO,
    CCR,
    CE_SMOTE,
    CURE_SMOTE,
    DBSMOTE,
    DEAGO,
    DE_oversampling,
    DSMOTE,
    DSRBF,
    E_SMOTE,
    Edge_Det_SMOTE,
    GASMOTE,
    G_SMOTE,
    Gaussian_SMOTE,
    Gazzah,
    IPADE_ID,
    ISMOTE,
    ISOMAP_Hybrid,
    KernelADASYN,
    LLE_SMOTE,
    LN_SMOTE,
    LVQ_SMOTE,
    Lee,
    MCT,
    MDO,
    MOT2LD,
    MSMOTE,
    MSYN,
    MWMOTE,
    NDO_sampling,
    NEATER,
    NRAS,
    NRSBoundary_SMOTE,
    NT_SMOTE,
    NoSMOTE,
    OUPS,
    PDFOS,
    ProWSyn,
    ROSE,
    RWO_sampling,
    Random_SMOTE,
    SDSMOTE,
    SL_graph_SMOTE,
    SMMO,
    SMOBD,
    SMOTE,
    SMOTEWB,
    SMOTE_AMSR,
    SMOTE_Cosine,
    SMOTE_D,
    SMOTE_ENN,
    SMOTE_FRST_2T,
    SMOTE_IPF,
    SMOTE_OUT,
    SMOTE_PSO,
    SMOTE_PSOBAT,
    SMOTE_RSB,
    SMOTE_TomekLinks,
    SN_SMOTE,
    SOI_CJ,
    SOMO,
    SPY,
    SSO,
    SUNDO,
    SVM_balance,
    SYMPROD,
    Safe_Level_SMOTE,
    Selected_SMOTE,
    Stefanowski,
    Supervised_SMOTE,
    TRIM_SMOTE,
    VIS_RST,
    V_SYNTH,
    cluster_SMOTE,
    distance_SMOTE,
    kmeans_SMOTE,
    polynom_fit_SMOTE_bus,
    polynom_fit_SMOTE_mesh,
    polynom_fit_SMOTE_poly,
    polynom_fit_SMOTE_star,
]


# Disable sv logging
import logging

logger = logging.getLogger("smote_variants")
logger.disabled = True

# Define CV - use postal-code base random state
from sklearn.model_selection import RepeatedStratifiedKFold

RSKF = RepeatedStratifiedKFold(n_repeats=2, n_splits=5, random_state=50372)

# Datasets
from collections import OrderedDict
from keel import find_datasets, parse_keel_dat

DS_NAMES = [ds_fname.split('.')[0] for ds_fname in os.listdir('datasets/keel')]

DS_NAMES = [
    "vehicle1",
    "segment0",
    "yeast-0-2-5-6_vs_3-7-8-9",
    "cleveland-0_vs_4",
    "ecoli4",
    "glass-0-1-6_vs_5",
    "abalone-21_vs_8",
    "poker-8-9_vs_5",
]


DATASETS = {
    relation: {'data': data, 'target': target} for relation, data, target in
    [parse_keel_dat(os.path.join('datasets', 'keel', f"{ds_name}.dat")) for ds_name in DS_NAMES]
}

# # Replace for UCI experiments
# from imblearn.datasets import fetch_datasets
# DS_NAMES = ['thyroid_sick']
# DATASETS = fetch_datasets(data_home="datasets", verbose=True)
# DATASETS = OrderedDict({k: v for k, v in DATASETS.items() if k in DS_NAMES})

from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier

CLASSIFIERS = [
    KNeighborsClassifier,
    # GaussianNB,
    # DecisionTreeClassifier,
    # SVC,
    # MLPClassifier,
]
