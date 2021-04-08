# %%
# Imports and reproducibility
# ---------------------------
#
# Set seed for reproducibility.

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

num_epochs = 50
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
cluster = KMeans(n_clusters=8, random_state=SEED).fit_predict(encoded_test)
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=cluster)
plt.show()


# %%
# Cluster analysis
# ----------------

test_info['cluster'] = cluster

_, ax0 = plt.subplots(3, 1, figsize=(5, 15))
test_info.boxplot('age', 'cluster', ax=ax0[0], showmeans=True)
test_info.boxplot('gender', 'cluster', ax=ax0[1], showmeans=True)
test_info.boxplot('weight', 'cluster', ax=ax0[2], showmeans=True)
plt.show()


test_data['cluster'] = cluster

_, ax1 = plt.subplots(5, 2, figsize=(10, 25))

features = ["bleeding", "plt", "shock", "haematocrit_percent", "bleeding_gum",
            "abdominal_pain", "ascites", "bleeding_mucosal", "bleeding_skin",
            "body_temperature"]

for i, feat in enumerate(features):
    test_data.boxplot(feat, 'cluster', ax=ax1[i % 5][i % 2], showmeans=True)

plt.show()

