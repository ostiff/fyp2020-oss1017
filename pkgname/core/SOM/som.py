import pandas as pd
from minisom import MiniSom
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from pkgname.utils.som_utils import *

def main():
    path = '../../../data/daily-profile.csv'
    SEED = 0
    print("Loading dataset")

    GENERAL_COLS = ["_uid", "dateResult", "GenderID", "patient_age", "covid_confirmed"]

    BONE_features = ["GLOB","TP","CALC","CALCOR","ALP","PHOS","ALB"]

    FBC_features = ["EOS", "MONO", "BASO", "NEUT",
                    "RBC", "WBC", "MCHC", "MCV",
                    "LY", "HCT", "RDW", "HGB",
                    "MCH", "PLT", "MPV", "NRBCA"]

    FBC_remove = ["WBC", "HGB", "HCT", "MCV"]   # remove because they have
                                                # correlations to other columns of over 0.9

    panel_features = [item for item in FBC_features if item not in FBC_remove]

    df = pd.read_csv(path, usecols=panel_features+GENERAL_COLS)
    df = df.dropna()
    df = df.drop_duplicates(subset='_uid', keep="first")
    df = df.drop(columns=["_uid", "GenderID", "dateResult", "covid_confirmed"])
    x = df.values #returns a numpy array
    scaler = preprocessing.MinMaxScaler()
    #scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    som = MiniSom(20, 20, x.shape[1],
        topology='hexagonal',
        activation_distance='euclidean',
        neighborhood_function='gaussian',
        sigma=5, learning_rate=.05,
        random_seed=SEED)

    # Train
    som.pca_weights_init(x)
    som.train_random(x, 10000000, verbose=True)


    diff_graph_hex(som, show=True, printout=False)
    feature_maps(som, cols=3, show=True, printout=False)

if __name__ == '__main__':
    main()


