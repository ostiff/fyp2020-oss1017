"""
Plot Utils for MiniSom
======================


"""

# Generic libraries
import pandas as pd
import numpy as np

# Other
import matplotlib.pyplot as plt
from matplotlib import rcParams
from minisom import MiniSom
from sklearn.datasets import make_blobs
from sklearn import preprocessing

# Utils
from pkgname.utils.som_utils import diff_graph_hex, feature_maps, feature_map

# Configuration
rcParams.update({'figure.autolayout': True})


# ----------------------
# Create dataset
# ----------------------
SEED = 0
N_CLUSTERS = 6
N_SAMPLES = N_CLUSTERS * 100
N_FEATURES = 16

# Show information
print("The dataset has {dim} dimensions and {clusters} clusters."
      .format(dim=N_FEATURES, clusters=N_CLUSTERS))

# Create dataset
x, y = make_blobs(n_samples=N_SAMPLES,
                  n_features=N_FEATURES,
                  centers=N_CLUSTERS,
                  cluster_std=3,
                  random_state=SEED)

# Scale data
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)

# ----------------------
# Train SOM
# ----------------------
# Create SOM
som = MiniSom(20, 20, x.shape[1],
    topology='hexagonal',
    activation_distance='euclidean',
    neighborhood_function='gaussian',
    sigma=2.5, learning_rate=.5,
    random_seed=SEED)

# Train
som.pca_weights_init(x)
som.train_random(x, 1000, verbose=True)

diff_graph_hex(som, show=True, printout=False)
feature_map(som, colnum=7, show=True, printout=False)
feature_maps(som, cols=4, show=True, printout=False)
feature_maps(som, cols=3, show=True, printout=False)


