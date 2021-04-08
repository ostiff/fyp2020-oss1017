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
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from definitions import DATA_DIR
from pkgname.core.VAE.vae import VAE, train_vae, plot_vae_loss, get_device, set_seed

SEED = 0

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
beta = 0.1

# Additional parameters
input_size = 10
h_dim = 5


# %%
# Load data
# ---------
#
# ``dengue.csv`` is a pre-processed version of the main dataset.
#
df = pd.read_csv(os.path.join(DATA_DIR, 'dengue.csv'))

data = ["bleeding", "plt", "shock", "haematocrit_percent",
        "bleeding_gum", "abdominal_pain", "ascites",
        "bleeding_mucosal", "bleeding_skin", "body_temperature"]

info = ["date", "age", "gender", "weight"]

train, test = train_test_split(df, test_size=0.2, random_state=SEED)

train_data = train[data]
test_data = test[data]
train_info = train[info]
test_info = test[info]

scaler = preprocessing.MinMaxScaler().fit(train_data)

train_scaled = scaler.transform(train_data.to_numpy())
test_scaled = scaler.transform(test_data.to_numpy())

loader_train = DataLoader(train_scaled, batch_size, shuffle=True)
loader_test = DataLoader(test_scaled, batch_size, shuffle=False)


# %%
# Beta-VAE
# --------

model = VAE(device=device, latent_dim=latent_dim, input_size=input_size, h_dim=h_dim).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
losses = train_vae(model, optimizer, loader_train, loader_test, num_epochs, beta)

# Plot losses
plot_vae_loss(losses, show=True, printout=False)

# %%
# Clustering
# ----------

# Encode test set and plot in 2D (assumes latent_dim = 2)
encoded_test = model.encode_inputs(loader_test)
plt.scatter(encoded_test[:, 0], encoded_test[:, 1])
plt.show()

# Perform clustering on encoded inputs
cluster = KMeans(n_clusters=3, random_state=SEED).fit_predict(encoded_test)
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=cluster)
plt.show()


# %%
# Cluster analysis
# ----------------
#
# These attributes were not used to train the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    test_info['cluster'] = cluster

    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.05)

    for i, feat in enumerate(['age','gender','weight']):
        fig.add_trace(
            go.Box(x=test_info["cluster"].values, y=test_info[feat].values,
                   boxpoints='outliers', boxmean=True),
            row=i + 1, col=1
        )
        fig.update_xaxes(title_text="Cluster", row=i + 1, col=1)
        fig.update_yaxes(title_text=feat, row=i + 1, col=1)

    fig.update_layout(height=1800, title_text="Attributes not used in training", showlegend=False)
fig

#%%
# The following attributes were used to train the model.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    test_data['cluster'] = cluster

    features = ["bleeding", "plt", "shock", "haematocrit_percent", "bleeding_gum",
                "abdominal_pain", "ascites", "bleeding_mucosal", "bleeding_skin",
                "body_temperature"]

    cols = 2
    rows = 5
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.05, horizontal_spacing= 0.2)

    for i, feat in enumerate(features):
        fig.add_trace(
            go.Box(x=test_data["cluster"].values, y=test_data[feat].values,
                   boxpoints='outliers', boxmean=True),
            row=(i // cols) + 1, col=(i % cols) + 1
        )
        fig.update_xaxes(title_text="Cluster", row=(i // cols) + 1, col=(i % cols) + 1)
        fig.update_yaxes(title_text=feat, row=(i // cols) + 1, col=(i % cols) + 1)

    fig.update_layout(height=3000, title_text="Attributes used in training", showlegend=False)
fig
