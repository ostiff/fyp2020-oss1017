"""
Beta-VAE used for clustering 1
==============================

Type of Variational Auto-Encoder where the KL divergence term in the loss
function is weighted by a parameter `beta`.

Training attributes: `bleeding`, `plt`, `shock`, `haematocrit_percent`,
`bleeding_gum`, `abdominal_pain`, `ascites`, `bleeding_mucosal`,
`bleeding_skin`, `body_temperature`.

Attributes used in cluster comparison: `age`, `gender`, `weight`.


"""

# %%
# Imports and reproducibility
# ---------------------------
#
# Set seed for reproducibility.

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

from pkgname.core.VAE.vae import VAE, train_vae, plot_vae_loss, get_device, set_seed
from pkgname.utils.data_loader import load_dengue
from pkgname.utils.plot_utils import plotBox, formatTable
from pkgname.utils.log_utils import Logger

logger = Logger('VAE_Dengue')

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
beta = 0.2


# %%
# Load data
# ---------
#
# ``dengue.csv`` is a pre-processed version of the main dataset.
#

features = ["dsource","date", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

df = load_dengue(usecols=['study_no']+features)

for feat in features:
    df[feat] = df.groupby('study_no')[feat].ffill().bfill()

df = df.loc[df['age'] <= 18]
df = df.dropna()

df = df.groupby(by="study_no", dropna=False).agg(
    dsource=pd.NamedAgg(column="dsource", aggfunc="last"),
    date=pd.NamedAgg(column="date", aggfunc="last"),
    age=pd.NamedAgg(column="age", aggfunc="max"),
    gender=pd.NamedAgg(column="gender", aggfunc="first"),
    weight=pd.NamedAgg(column="weight", aggfunc=np.mean),
    bleeding=pd.NamedAgg(column="bleeding", aggfunc="max"),
    plt=pd.NamedAgg(column="plt", aggfunc="min"),
    shock=pd.NamedAgg(column="shock", aggfunc="max"),
    haematocrit_percent=pd.NamedAgg(column="haematocrit_percent", aggfunc="max"),
    bleeding_gum=pd.NamedAgg(column="bleeding_gum", aggfunc="max"),
    abdominal_pain=pd.NamedAgg(column="abdominal_pain", aggfunc="max"),
    ascites=pd.NamedAgg(column="ascites", aggfunc="max"),
    bleeding_mucosal=pd.NamedAgg(column="bleeding_mucosal", aggfunc="max"),
    bleeding_skin=pd.NamedAgg(column="bleeding_skin", aggfunc="max"),
    body_temperature=pd.NamedAgg(column="body_temperature", aggfunc=np.mean),
).dropna()

mapping = {'Female': 0, 'Male': 1}
df = df.replace({'gender': mapping})

info_feat = ["dsource", "age", "gender", "weight"]
data_feat = ["bleeding", "plt", "shock", "haematocrit_percent", "bleeding_gum",
             "abdominal_pain", "ascites", "bleeding_mucosal", "bleeding_skin",
             "body_temperature"]

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
h_dim=5
model = VAE(device=device, latent_dim=latent_dim, input_size=input_size, h_dim=h_dim).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
losses = train_vae(model, optimizer, loader_train, loader_test, num_epochs, beta)

# Save model
logger.save_object(model)

# Plot losses
plot = plot_vae_loss(losses, show=True, printout=False)
logger.add_plt(plot)

# %%
# Clustering
# ----------

colours = ["red", "blue", "limegreen", "orangered", "yellow",
           "violet", "salmon", "slategrey", "green", "crimson"][:N_CLUSTERS]

# Encode test set and plot in 2D (assumes latent_dim = 2)
encoded_test = model.encode_inputs(loader_test)
plt.scatter(encoded_test[:, 0], encoded_test[:, 1])
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
# Cluster analysis
# ----------------
#
# Table
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    mapping = {0: 'Female', 1: 'Male'}
    table_df = test.replace({'gender': mapping})

    table_df['cluster'] = cluster

columns = info_feat+data_feat
nonnormal = list(table_df[columns].select_dtypes(include='number').columns)
categorical = list(set(columns).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)

rename = {'haematocrit_percent': 'hct',
          'body_temperature': 'temperature'}

table = TableOne(table_df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', rename=rename, missing=False)

html = formatTable(table, colours, labels)
logger.append_html(html.render())
html

# %%
# These attributes were not used to train the model.

fig, html = plotBox(data=test_info,
                    features=info_feat,
                    clusters=cluster,
                    colours=colours,
                    title="Attributes not used in training",
                    #path="a.html"
                    )
logger.append_html(html)
fig


#%%
# The following attributes were used to train the model.

fig, html = plotBox(data=test_data,
                    features=data_feat,
                    clusters=cluster,
                    colours=colours,
                    title="Attributes used in training",
                    #path="b.html"
                    )
logger.append_html(html)
fig

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
        'beta': beta,
        'input_size':input_size,
        'h_dim':h_dim,
        'features': features,
        'info_feat': info_feat,
        'data_feat': data_feat
    }
)

logger.create_report()