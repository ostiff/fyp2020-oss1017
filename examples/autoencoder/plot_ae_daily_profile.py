"""
Simple autoencoder for pathology data clustering
==================================================

Training attributes: FBC panel

Attributes used in cluster comparison: ...
"""

import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tableone import TableOne

from pkgname.core.AE.autoencoder import Autoencoder, train_autoencoder, plot_autoencoder_loss, get_device, set_seed
from pkgname.utils.data_loader import load_pathology
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger

logger = Logger('AE_Pathology', enable=True, compress=False)

SEED = 0
N_CLUSTERS = 3

# Set seed
set_seed(SEED)

# Get device
device = get_device(usegpu=False)

num_epochs = 100
learning_rate = 0.00005
batch_size = 16
latent_dim = 2


df = load_pathology(usedefault=True, dropna=True)
del df["_uid"]
del df["dateResult"]
info_feat = ["GenderID", "patient_age", "covid_confirmed"]
data_feat = ["EOS", "MONO", "BASO", "NEUT",
             "RBC", "WBC", "MCHC", "MCV",
             "LY", "HCT", "RDW", "HGB",
             "MCH", "PLT", "MPV", "NRBCA"]


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


# Additional parameters
input_size = len(data_feat)
layers=[10,4]
model = Autoencoder(input_size=input_size,
                    layers=layers,
                    latent_dim=latent_dim,
                    device=device).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
losses = train_autoencoder(model, optimizer, loader_train, loader_test, num_epochs)

# Save model
logger.save_object(model)

# Plot losses
plot = plot_autoencoder_loss(losses, show=True, printout=False)
logger.add_plt(plot)

# %%
#

colours = colours[:N_CLUSTERS]

# Encode test set and plot in 2D (assumes latent_dim = 2)
encoded_test = model.encode_inputs(loader_test)
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=test_info['covid_confirmed'])
logger.add_plt(plt.gcf())
plt.show()


# Perform clustering on encoded inputs
cluster = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(encoded_test)

labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

scatter = plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=cluster, cmap=ListedColormap(colours))
plt.legend(handles=scatter.legend_elements()[0], labels=labels)
logger.add_plt(plt.gcf())
plt.show()

# %%
#

# Table
with warnings.catch_warnings():
    test['cluster'] = cluster

columns = info_feat+data_feat
nonnormal = list(test[columns].select_dtypes(include='number').columns)
nonnormal.remove('GenderID')
categorical = list(set(columns).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)



table = TableOne(test, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', missing=False)

html = formatTable(table, colours, labels)
logger.append_html(html.render())
html

# %%
#

fig, html = plotBox(data=test_info,
                    features=info_feat,
                    clusters=cluster,
                    colours=colours,
                    title="Attributes not used in training",
                    )
logger.append_html(html)
fig

# %%
#

fig, html = plotBox(data=test_data,
                    features=data_feat,
                    clusters=cluster,
                    colours=colours,
                    title="Attributes used in training",
                    )
logger.append_html(html)
fig

# %%
# Logging
# -------

# Log parameters
logger.save_parameters(
    {
        'SEED': SEED,
        'N_CLUSTERS': N_CLUSTERS,
        'device': str(device),
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'input_size':input_size,
        'layers':layers,
        'info_feat': info_feat,
        'data_feat': data_feat
    }
)

logger.create_report()
