"""
Dimensionality reduction analysis
=================================

Metrics used to analyse results of dimensionality reduction techniques.

"""


import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt

from pkgname.core.AE.autoencoder import get_device, set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule
from definitions import ROOT_DIR

# --------------
# Load data
# --------------

SEED = 0
np.random.seed(SEED)

batch_size = 16
N_points = 1000
MODEL_PATH = os.path.join(ROOT_DIR, 'examples', 'autoencoder', 'model')
TSNE_POINTS_PATH = os.path.join(ROOT_DIR, 'examples', 'result_analysis', 'points', 'tsne_embedded')
SOM_POINTS_PATH = os.path.join(ROOT_DIR, 'examples', 'result_analysis', 'points', 'som_embedded')

# Set seed
set_seed(SEED)

# Get device
device = get_device(False)

features = ["dsource", "date", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]
info_feat = ["dsource", "shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
             "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

df = load_dengue(usecols=['study_no'] + features)
df = df.loc[df['age'] <= 18]

df = df.dropna(subset=data_feat + ['date'])

for feat in features:
    df[feat] = df.groupby('study_no')[feat].ffill().bfill()

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

df = IQR_rule(df, ['plt'])
data = df[data_feat]

scaler = preprocessing.MinMaxScaler().fit(data)

# Points to sample
ind_list = np.random.choice(len(df.index), N_points)
df = df.iloc[ind_list]

data = df[data_feat]
info = df[info_feat]

scaled = scaler.transform(data)
loader = DataLoader(scaled, batch_size, shuffle=False)


# %%
# AE
# -------
# Results show distances between points are relatively well preserved and points
# in the latent space

model = pickle.load(open(MODEL_PATH, 'rb'))
encoded_ae = model.encode_inputs(loader)

og_dist = cdist(scaled, scaled).flatten()
ae_dist = cdist(encoded_ae, encoded_ae).flatten()

plt.scatter(ae_dist, og_dist, s=1)
plt.xlabel("Encoded points distance")
plt.ylabel("Original distance (scaled data)")
plt.title("Sheppard diagram (AE)")
plt.show()

# %%
#

plt.scatter(encoded_ae[:,0], encoded_ae[:,1], c=info['shock'])
plt.xlabel("x")
plt.ylabel("y")
plt.title("AE reduced data (shock)")
plt.show()

corr = pearsonr(og_dist, ae_dist)
print(f'Corr AE distances: {corr[0]}; p-val: {corr[1]}')

padded_ae = np.c_[encoded_ae, np.zeros(N_points), np.zeros(N_points), np.zeros(N_points) ]
proc = procrustes(scaled, padded_ae)
print(f'AE procrustes: {proc[2]}')


# %%
# PCA
# -------

pca = PCA(n_components=2).fit(scaled)
pca_points = pca.transform(scaled)

pca_dist = cdist(pca_points, pca_points).flatten()

plt.scatter(pca_dist, og_dist, s=1)
plt.xlabel("Encoded points distance")
plt.ylabel("Original distance (scaled data)")
plt.title("Sheppard diagram (PCA)")
plt.show()

# %%
#

plt.scatter(pca_points[:,0], pca_points[:,1], c=info['shock'])
plt.xlabel("x")
plt.ylabel("y")
plt.title("PCA reduced data (shock)")
plt.show()

corr = pearsonr(og_dist, pca_dist)
print(f'Corr PCA distances: {corr[0]}; p-val: {corr[1]}')

padded_pca = np.c_[pca_points, np.zeros(N_points), np.zeros(N_points), np.zeros(N_points) ]
proc = procrustes(scaled, padded_pca)
print(f'PCA procrustes: {proc[2]}')


# %%
# t-SNE
# -------

tsne_points = np.take(pickle.load(open(TSNE_POINTS_PATH, 'rb')), ind_list, axis=0)

tsne_dist = cdist(tsne_points, tsne_points).flatten()

plt.scatter(tsne_dist, og_dist, s=1)
plt.xlabel("Encoded points distance")
plt.ylabel("Original distance (scaled data)")
plt.title("Sheppard diagram (t-SNE)")
plt.show()

# %%
#

plt.scatter(tsne_points[:,0], tsne_points[:,1], c=info['shock'])
plt.xlabel("x")
plt.ylabel("y")
plt.title("t-SNE reduced data (shock)")
plt.show()

corr = pearsonr(og_dist, tsne_dist)
print(f'Corr t-SNE distances: {corr[0]}; p-val: {corr[1]}')

padded_tsne = np.c_[tsne_points, np.zeros(N_points), np.zeros(N_points), np.zeros(N_points) ]
proc = procrustes(scaled, padded_tsne)
print(f'TSNE procrustes: {proc[2]}')


# %%
# SOM
# -------

som_points = np.take(pickle.load(open(SOM_POINTS_PATH, 'rb')), ind_list, axis=0)

som_dist = cdist(som_points, som_points).flatten()

plt.scatter(som_dist, og_dist, s=1)
plt.xlabel("Encoded points distance")
plt.ylabel("Original distance (scaled data)")
plt.title("Sheppard diagram (SOM)")
plt.show()

# %%
#

plt.scatter(som_points[:,0], som_points[:,1], c=info['shock'])
plt.xlabel("x")
plt.ylabel("y")
plt.title("SOM reduced data (shock)")
plt.show()

corr = pearsonr(og_dist, som_dist)
print(f'Corr SOM distances: {corr[0]}; p-val: {corr[1]}')

padded_som = np.c_[som_points, np.zeros(N_points), np.zeros(N_points), np.zeros(N_points) ]
proc = procrustes(scaled, padded_som)
print(f'SOM procrustes: {proc[2]}')