"""
SOM Dimensionality Reduction (II)
=================================

This code is equivalent to :ref:`sphx_glr__examples_som_plot_som_sample_01.py`
but uses the library ``minisom``. Note that the visualisation produced is not as
nice as ``SimpSOM``. Yet this can achieve through ``matplotlib`` or ``seaborn``.

See https://github.com/JustGlowing/minisom/blob/master/examples

.. warning:: ``minisom`` is computationally more efficient than ``SimpSOM``
"""

# Generic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Other
from matplotlib import rcParams
from minisom import MiniSom
from sklearn.datasets import make_blobs
from sklearn import preprocessing

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
    neighborhood_function='gaussian',
    sigma=1.5, learning_rate=.5,
    random_seed=SEED)

# Train
som.pca_weights_init(x)
som.train_random(x, 1000, verbose=True)

# Compute projections
prj = np.array([som.winner(e) for e  in x])

# Get weights
W = som.get_weights()

# -------------
# Plot overall
# -------------
# Create figure
f, ax = plt.subplots(1, 1, figsize=(6, 6))

# Plot
cs = plt.pcolor(som.distance_map().T)
plt.scatter(prj[:, 0], prj[:, 1], c=y, cmap='rainbow')
ax.set(title='Nodes Grid w Weights Difference',
       xticks=[], yticks=[], aspect='equal')

# Add colorbar
cbar = f.colorbar(cs, shrink=0.78)

# -------------------
# Plot feature planes
# -------------------
# Create figure
f, axes = plt.subplots(4, 4, figsize=(8,8))
axes = axes.flatten()

# Plot feature planes
for i, f in enumerate(range(N_FEATURES)):
    axes[i].pcolor(W[:, :, f].T)
    axes[i].set(title='Feature %s' % f, aspect='equal',
        yticks=[], xticks=[])

# Set axes
plt.suptitle("Node Grid w Feature #i")
plt.tight_layout()

# Show
plt.show()