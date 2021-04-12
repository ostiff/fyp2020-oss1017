"""
Beta-VAE used for clustering
============================

Type of Variational Auto-Encoder where the KL divergence term in the loss
function is weighted by a parameter `beta`.

"""

# %%
# Imports and reproducibility
# ---------------------------
#
# Set seed for reproducibility.

import os
import warnings
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from definitions import DATA_DIR
from pkgname.core.VAE.vae import VAE, train_vae, plot_vae_loss, get_device, set_seed

SEED = 0
N_CLUSTERS = 3

# Set seed
set_seed(SEED)

# Get device
device = get_device(False)

# %%
# Hyperparameters
# ---------------

num_epochs = 20
learning_rate = 0.0001
batch_size = 16
latent_dim = 2
beta = 0


# %%
# Load data
# ---------
#
# ``dengue.csv`` is a pre-processed version of the main dataset.
#
df = pd.read_csv(os.path.join(DATA_DIR, 'dengue.csv'))

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin", ]

data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

train, test = train_test_split(df, test_size=0.2, random_state=SEED)

train_data = train[data_feat]
test_data = test[data_feat]
train_info = train[info_feat]
test_info = test[info_feat]

scaler = preprocessing.MinMaxScaler().fit(train_data)

train_scaled = scaler.transform(train_data.to_numpy())
test_scaled = scaler.transform(test_data.to_numpy())

loader_train = DataLoader(train_scaled, batch_size, shuffle=True)
loader_test = DataLoader(test_scaled, batch_size, shuffle=False)


# %%
# Beta-VAE
# --------

# Additional parameters
input_size = len(train_data.columns)
layers=[15,10,5]
model = VAE(device=device, latent_dim=latent_dim, input_size=input_size, layers=layers).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
losses = train_vae(model, optimizer, loader_train, loader_test, num_epochs, beta)

# Plot losses
plot_vae_loss(losses, show=True, printout=False)

# %%
# Clustering
# ----------

colours = ["red", "blue", "limegreen", "orangered", "yellow",
           "violet", "salmon", "slategrey", "green", "crimson"][:N_CLUSTERS]

# Encode test set and plot in 2D (assumes latent_dim = 2)
encoded_test = model.encode_inputs(loader_test)
plt.scatter(encoded_test[:, 0], encoded_test[:, 1])
plt.show()

# Perform clustering on encoded inputs
cluster = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(encoded_test)
scatter = plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=cluster, cmap=ListedColormap(colours))
plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(N_CLUSTERS)])
plt.show()


# %%
# Cluster analysis
# ----------------
#
# These attributes were not used to train the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    test_info['cluster'] = cluster

    cols = 2
    rows = (len(info_feat) + 1) // 2
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.05)

    for i, feat in enumerate(info_feat):
        for j in range(N_CLUSTERS):
            fig.add_trace(
                go.Box(
                    y=test_info[test_info['cluster'] == j][feat].values,
                    boxpoints='outliers', boxmean=True, name=f"Cluster {j}",
                    marker=dict(color=colours[j]),
                ),
                row = (i // cols) + 1, col = (i % cols) + 1
            )
        fig.update_yaxes(title_text=feat, row=(i // cols) + 1, col=(i % cols) + 1)

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=477*rows, title_text="Attributes not used in training", showlegend=False)
fig

#%%
# The following attributes were used to train the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    test_data['cluster'] = cluster

    cols = 2
    rows =  (len(data_feat) + 1) // 2
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.05, horizontal_spacing= 0.2)

    for i, feat in enumerate(data_feat):
        for j in range(N_CLUSTERS):
            fig.add_trace(
                go.Box(
                    y=test_data[test_data['cluster'] == j][feat].values,
                    boxpoints='outliers', boxmean=True, name=f"Cluster {j}",
                    marker=dict(color=colours[j]),
                ),
                row = (i // cols) + 1, col = (i % cols) + 1
            )
        fig.update_yaxes(title_text=feat, row=(i // cols) + 1, col=(i % cols) + 1)

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=477*rows, title_text="Attributes used in training", showlegend=False)

fig
