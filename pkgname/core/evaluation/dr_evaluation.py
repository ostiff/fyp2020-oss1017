"""
Dimensionality reduction evaluation functions.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
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
mpl.rcParams.update({'font.family': 'serif'})


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

    return areas[1] / areas[0]


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


def distance_metrics(original_data, reduced_data, n_points, method_name, verbose=True):

    # Points to sample
    ind_list = np.random.choice(len(original_data), n_points)
    original_data = np.take(original_data, ind_list, axis=0)
    reduced_data = np.take(reduced_data, ind_list, axis=0)

    if verbose:
        print('#########################################')
        print(f'# Distance metrics: {method_name}')

    original_dist = cdist(original_data, original_data).flatten()
    new_dist = cdist(reduced_data, reduced_data).flatten()

    # Sheppard Diagram
    fig = plt.figure()
    sns.scatterplot(x=new_dist, y=original_dist, color=".3", linewidth=0, s=1)
    plt.xlabel("Encoded points distance")
    plt.ylabel("Original distance (scaled data)")

    # Correlation old vs new distances
    pearson = pearsonr(original_dist, new_dist)
    spearman = spearmanr(original_dist, new_dist)

    # Procrustes
    padded_data = np.c_[reduced_data, np.zeros(n_points), np.zeros(n_points), np.zeros(n_points)]
    try:
        proc = procrustes(original_data, padded_data)
    except ValueError as e:
        print(str(e))
        proc = str(e)
    if verbose:
        print(f'pearson {method_name} distances: {pearson[0]}; p-val: {pearson[1]}')
        print(f'spearman {method_name} distances: {spearman[0]}; p-val: {spearman[1]}')
        print(f'Procrustes {method_name}: {proc[2]}')

    return {
        'procrustes': proc[2],
        'spearman':spearman,
        'pearson': pearson,
    }, fig


def density_metrics(info_df, reduced_data, labels, method_name, colours=None, scale=False):
    results = {}

    if colours is None:
        colours = np.array(color_palette('pastel', 2).as_hex())

    outline_colours = [adjust_lightness(c, 0.5) for c in colours]

    if scale:
        reduced_data = preprocessing.StandardScaler().fit_transform(reduced_data)

    print('#########################################')
    print(f'# Density metrics: {method_name}')

    rows = len(labels)
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), dpi=300)
    fig.tight_layout()
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < 3 * rows:
            ax.axis('equal')
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

    for i, label_name in enumerate(labels):

        print(f'{label_name}:')
        y = info_df[label_name].to_numpy().astype(int)

        # GMM with 1 component
        try:
            plot = plt.axes(axes[i*3 + 0])
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colours[y], s=8)
            plt.title(f"{method_name} gmm ratio ({label_name})")

            gmm_all = mixture.BayesianGaussianMixture(n_components=1, covariance_type='full').fit(reduced_data)
            gmm_attr = mixture.BayesianGaussianMixture(n_components=1, covariance_type='full').fit(reduced_data[y == 1])
            means = np.concatenate([gmm_all.means_, gmm_attr.means_])
            covs = np.concatenate([gmm_all.covariances_, gmm_attr.covariances_])
            gmm_ratio = plot_ellipse(means, covs, plot, outline_colours)

            print(f'Gaussian area ratio: {gmm_ratio}')
            results[f'{label_name}_gmm_ratio'] = gmm_ratio
        except Exception as e:
            results[f'{label_name}_gmm_ratio'] = str(e)


        # Convex hulls ratio
        try:
            plt.axes(axes[i*3 + 1])
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colours[y], s=8)
            plt.title(f"{method_name} convex hulls ({label_name})")

            hull_all = plot_polygon(reduced_data, 0.0, outline_colours[0])
            hull_atr = plot_polygon(reduced_data[y == 1], 0.0, outline_colours[1])

            print(f'Convex hull area ratio: {hull_atr / hull_all}')
            results[f'{label_name}_convex_hull_ratio'] = hull_atr / hull_all
        except Exception as e:
            results[f'{label_name}_convex_hull_ratio'] = str(e)


        # Concave hulls ratio
        try:
            plt.axes(axes[i*3 + 2])
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colours[y], s=8)
            plt.title(f"{method_name} concave hulls ({label_name})")
            hull_all = plot_polygon(reduced_data, 5.0, outline_colours[0])
            hull_atr = plot_polygon(reduced_data[y == 1], 5.0, outline_colours[1])

            print(f'Concave hull area ratio: {hull_atr / hull_all}')
            results[f'{label_name}_concave_hull_ratio'] = hull_atr / hull_all
        except Exception as e:
            results[f'{label_name}_concave_hull_ratio'] = str(e)
    print()

    return results, fig


def main():
    # --------------
    # Load data
    # --------------

    SEED = 0
    np.random.seed(SEED)

    batch_size = 16
    MODEL_PATH = os.path.join(ROOT_DIR, 'examples', 'autoencoder', 'model')

    # Set seed
    set_seed(SEED)
    np.random.seed(SEED)

    features = ["dsource", "date", "age", "gender", "weight", "bleeding", "plt",
                "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

    df = load_dengue(usecols=['study_no'] + features)

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
    df = df.replace({'gender': mapping})

    info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                 "bleeding_mucosal", "bleeding_skin", "gender"]
    data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

    info = df[info_feat]
    data = df[data_feat]

    scaler = preprocessing.MinMaxScaler().fit(data)

    scaled = scaler.transform(data)
    loader = DataLoader(scaled, batch_size, shuffle=False)

    # AE eval metrics

    model = pickle.load(open(MODEL_PATH, 'rb'))
    encoded_ae = model.encode_inputs(loader)

    res, fig = distance_metrics(scaled, encoded_ae, n_points=1000, method_name='AE', verbose=True)
    plt.show()
    print(res)
    res, fig = density_metrics(info, encoded_ae, ['shock', 'bleeding', 'ascites'], 'AE')
    plt.show()
    print(res)

if __name__ == '__main__':
    main()
