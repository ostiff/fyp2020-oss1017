"""
SOM Dimensionality Reduction
============================

Using ``SimpSOM`` to reduce the dimensionality of a synthetic dataset.
Clustering performed using k-means.

"""
import pandas as pd
import SimpSOM as sps
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

SEED = 0

# Create sample dataset
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


# Initialise SOM and train
net = sps.somNet(20, 20, x, PBC=False)
net.train(0.01, 20000)

# Plot SOM
net.diff_graph()
plt.show()

# Project datapoints onto SOM and extract positions
prj=np.array(net.project(x))

# Apply k-means clustering to dimensionality reduced data
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit(prj)

# Plot clustering results
plt.scatter(prj[:,0],prj[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()

data_labels = pd.Series(y, name='Class')
cluster_labels = pd.Series([chr(ord(str(num))+17) for num in kmeans.labels_], name='Cluster')
print(pd.crosstab(data_labels, cluster_labels, dropna=False))

# Plot individual feature maps
for i in range(N_FEATURES):
    net.nodes_graph(colnum=i, printout=False)
    plt.show()
