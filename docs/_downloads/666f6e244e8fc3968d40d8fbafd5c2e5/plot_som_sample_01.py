"""
SOM Dimensionality Reduction
============================

Use ``SimpSOM`` to reduce the dimensionality of a synthetic dataset.

Clustering performed using k-means.

"""
from pkgname.utils.print_utils import suppress_stdout

import SimpSOM as sps
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})



SEED = 0

# %%
# Synthetic Dataset
# -------------------
#
# Dataset created using ``sklearn.datasets.make_blobs``.
#

N_CLUSTERS = 6
N_SAMPLES = N_CLUSTERS * 100
N_FEATURES = 16

print("The dataset has {dim} dimensions and {clusters} clusters.".format(dim=N_FEATURES, clusters=N_CLUSTERS))

x, y = make_blobs(n_samples=N_SAMPLES,
                  n_features=N_FEATURES,
                  centers=N_CLUSTERS,
                  cluster_std=3,
                  random_state=SEED)

# Scale data
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)

# %%
# Self Organising Map
# -------------------
#
# Initialise and train model using the synthetic dataset.
# Display node weight difference graph.
#

with suppress_stdout():
    net = sps.somNet(20, 20, x, PBC=False)
    net.train(0.01, 20000)

net.diff_graph(printout=False, show=True)


# %%
# k-Means Clustering
# -------------------
#
# Project datapoints onto the map and extract their coordinates
# to use them in 2D k-Means clustering.
#

prj=np.array(net.project(x, printout=False))

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit(prj)

plt.scatter(prj[:,0],prj[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()


# %%
# Clustering Results
# -------------------
#
# Compare dataset class labels to obtained clusters.
#
#
data_labels = pd.Series(y, name='Class')
cluster_labels = pd.Series([chr(ord(str(num))+17) for num in kmeans.labels_], name='Cluster')
print(pd.crosstab(data_labels, cluster_labels, dropna=False))


# %%
# SOM individual feature maps
# ----------------------------
#
# Display feature maps for all features in the original dataset.
#
#
for i in range(N_FEATURES):
    net.nodes_graph(colnum=i, printout=False, show=True)
