"""
Plot SOM w/ day labels for one patient
======================================

.................
"""

import pandas as pd
from minisom import MiniSom
from sklearn import preprocessing
import datetime as dt

from pkgname.utils.som_utils import *

path = '../../data/daily-profile.csv'
SEED = 0

TIME_PERIODS = 4

GENERAL_COLS = ["_uid", "dateResult", "GenderID", "patient_age", "covid_confirmed"]

FBC_features = ["EOS", "MONO", "BASO", "NEUT",
                "RBC", "WBC", "MCHC", "MCV",
                "LY", "HCT", "RDW", "HGB",
                "MCH", "PLT", "MPV", "NRBCA"]

FBC_remove = ["WBC", "HGB", "HCT", "MCV"]   # remove because they have
                                            # correlations to other columns of over 0.9

panel_features = [item for item in FBC_features if item not in FBC_remove]

df = pd.read_csv(path, usecols=panel_features+GENERAL_COLS)
df = df.dropna()

# Pick a patient at random and label the different days on the SOM
pid = df.sample(n=1).iloc[0]['_uid']
patient_df = df.loc[df['_uid'] == pid]

df = df.drop(columns=["_uid", "patient_age", "GenderID", "dateResult", "covid_confirmed"])

feature_names = df.columns

x = df.values #returns a numpy array
#scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

som = MiniSom(22, 22, x.shape[1],
    topology='hexagonal',
    activation_distance='euclidean',
    neighborhood_function='gaussian',
    sigma=2, learning_rate=.5,
    random_seed=SEED)

# Train
som.pca_weights_init(x)
som.train_random(x, 1000000, verbose=True)


diff_graph_hex(som, show=True, printout=False)
feature_maps(som, feature_names, cols=4, show=True, printout=False)

patient_df = patient_df.sort_values(by=['dateResult'])
p_data = patient_df.drop(columns=["_uid", "patient_age", "GenderID", "dateResult", "covid_confirmed"]).values
number_samples(som, p_data, show=True, printout=False)

