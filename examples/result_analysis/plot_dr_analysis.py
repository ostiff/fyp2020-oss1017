"""
Dimensionality reduction analysis
=================================

Metrics used to analyse results of dimensionality reduction techniques.

"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from scipy import linalg
import alphashape
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from scipy.stats import spearmanr, pearsonr
from sklearn import mixture
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from seaborn import color_palette
import matplotlib as mpl
import matplotlib.pyplot as plt

from pkgname.core.AE.autoencoder import get_device, set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import adjust_lightness
from definitions import ROOT_DIR

sys.path.insert(0, os.path.abspath('.'))
mpl.rcParams.update({'figure.autolayout': True})


def plot_ellipse(means, covariances, plot, colours):
    areas = []

    for i, (mean, covar, colour) in enumerate(zip(
            means, covariances, colours)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, edgecolor=colour, fill=False, lw=3, ls='--')
        ell.set_clip_box(plot.bbox)
        plot.add_artist(ell)

        areas.append(np.pi * v[0] * v[1] / 4)

    print(f'Gaussian area ratio: {areas[1] / areas[0]}')


def plot_polygon(points, alpha, colour):
    poly = alphashape.alphashape(points, alpha)

    if poly.geom_type == 'Polygon':
        plt.plot(*poly.exterior.xy, lw=3, ls='--', c=colour)
        area = poly.area
    else:
        if alpha > 0:
            area = plot_polygon(points, max(0, alpha - 0.5), colour)
        else:
            raise IOError('Shape is not a polygon.')

    return area


def eval_dim_reduction(original_points, original_dist, new_points, labels, label_name, method_name, colours):
    outline_colours = [adjust_lightness(c, 0.5) for c in colours]
    new_points = preprocessing.MinMaxScaler().fit_transform(new_points)

    print('#########################################')
    print(f'# {method_name}')

    new_dist = cdist(new_points, new_points).flatten()

    rows = 2
    cols = 2

    fig = plt.figure(dpi=400)

    # Sheppard Diagram
    plt.subplot(rows, cols, 1)
    plt.scatter(new_dist, original_dist, s=1)
    plt.xlabel("Encoded points distance")
    plt.ylabel("Original distance (scaled data)")
    plt.title(f"Sheppard diagram ({method_name})")

    # Correlation old vs new distances
    corr = pearsonr(original_dist, new_dist)
    print(f'Corr pearsonr{method_name} distances: {corr[0]}; p-val: {corr[1]}')
    corr = spearmanr(original_dist, new_dist)
    print(f'Corr spearmanr {method_name} distances: {corr[0]}; p-val: {corr[1]}')

    # Procrustes
    padded_data = np.c_[new_points, np.zeros(N_points), np.zeros(N_points), np.zeros(N_points)]
    proc = procrustes(original_points, padded_data)
    print(f'Procrustes {method_name}: {proc[2]}')

    # GMM with 1 component
    plot = plt.subplot(rows, cols, 2)
    plt.scatter(new_points[:, 0], new_points[:, 1], c=colours[labels], s=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method_name} gmm ratio ({label_name})")

    gmm_all = mixture.BayesianGaussianMixture(n_components=1, covariance_type='full').fit(new_points)
    gmm_attr = mixture.BayesianGaussianMixture(n_components=1, covariance_type='full').fit(new_points[labels == 1])
    means = np.concatenate([gmm_all.means_,gmm_attr.means_])
    covs = np.concatenate([gmm_all.covariances_,gmm_attr.covariances_])
    plot_ellipse(means, covs, plot, outline_colours)

    # Convex hulls ratio
    plt.subplot(rows, cols, 3)
    plt.scatter(new_points[:, 0], new_points[:, 1], c=colours[labels], s=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method_name} convex hulls ({label_name})")

    hull_all = plot_polygon(new_points, 0.0, outline_colours[0])
    hull_atr = plot_polygon(new_points[labels == 1], 0.0, outline_colours[1])

    print(f'Convex hull area ratio: {hull_atr / hull_all}')

    # Concave hulls ratio
    plt.subplot(rows, cols, 4)
    plt.scatter(new_points[:, 0], new_points[:, 1], c=colours[labels], s=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method_name} concave hulls ({label_name})")
    hull_all = plot_polygon(new_points, 5.0, outline_colours[0])
    hull_atr = plot_polygon(new_points[labels == 1], 5.0, outline_colours[1])

    print(f'Concave hull area ratio: {hull_atr / hull_all}')

    plt.show()

    print()


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

np.random.seed(SEED)
set_seed(SEED)

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

df = IQR_rule(df, ['plt'])

mapping = {'Female': 0, 'Male': 1}
before_mapping = df.copy()

df = df.replace({'gender': mapping})

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
             "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

info = df[info_feat]
data = df[data_feat]

scaler = preprocessing.MinMaxScaler().fit(data)

# Points to sample
ind_list = np.random.choice(len(df.index), N_points)
df = df.iloc[ind_list]

data = df[data_feat]
info = df[info_feat]

scaled = scaler.transform(data)
loader = DataLoader(scaled, batch_size, shuffle=False)
og_dist = cdist(scaled, scaled).flatten()

colours = np.array(color_palette('pastel', 2).as_hex())
labels = info['shock'].to_numpy().astype(int)


# %%
# AE
# -------
# Results show distances between points are relatively well preserved and points
# in the latent space

model = pickle.load(open(MODEL_PATH, 'rb'))
encoded_ae = model.encode_inputs(loader)

eval_dim_reduction(scaled, og_dist, encoded_ae, labels, 'shock', 'AE', colours)


# %%
# PCA
# -------

pca = PCA(n_components=2).fit(scaled)
pca_points = pca.transform(scaled)

eval_dim_reduction(scaled, og_dist, pca_points, labels, 'shock', 'PCA', colours)


# %%
# t-SNE
# -------

tsne_points = np.take(pickle.load(open(TSNE_POINTS_PATH, 'rb')), ind_list, axis=0)

eval_dim_reduction(scaled, og_dist, tsne_points, labels, 'shock', 't-SNE', colours)


# %%
# SOM
# -------

som_points = np.take(pickle.load(open(SOM_POINTS_PATH, 'rb')), ind_list, axis=0)

eval_dim_reduction(scaled, og_dist, som_points, labels, 'shock', 'SOM', colours)
