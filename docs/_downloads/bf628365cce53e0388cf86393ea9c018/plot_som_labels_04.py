"""
Plot SOM w/ Labelling
=====================

Project labels onto Node difference map.

The standard deviation of the clusters in the dataset was increased compared
to :ref:`sphx_glr__examples_som_plot_som_sample_01.py` to illustrate how nodes
can host datapoints with different labels.

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
from pkgname.utils.som_utils import diff_graph_hex, project_labels

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
                  cluster_std=6,
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

project_labels(som, x, y, show=True, printout=False)


