"""
t-SNE Experiment figures
========================

"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn import preprocessing
from seaborn import color_palette

from pkgname.core.evaluation.dr_evaluation import distance_metrics
from pkgname.utils.data_loader import load_dengue, IQR_rule
from definitions import ROOT_DIR
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.abspath('.'))

# mpl.use("pgf")
mpl.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    # 'font.family': 'serif',
    # 'text.usetex': True,
    # 'font.size': 18,

    'font.family': 'serif',
    'font.weight': 'light',
    'font.size': 16,
})

SEED = 0
DBSCAN_eps = 4
DBSCAN_min_samples = 10
np.random.seed(SEED)

TSNE_POINTS_PATH = os.path.join(ROOT_DIR, 'examples', 'report_figures', 'tsne_figures', 'perp_40')

features_labels = ["DSource", "Date", "Age", "Gender", "Weight", "Bleeding",
                   "Platelets", "Shock", "Haematocrit", "Bleeding gum", "Abdominal pain",
                   "Ascites", "Bleeding mucosal", "Bleeding skin", "Body temperature"]

features = ["dsource","date", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

label_mapping = dict(zip(features, features_labels))

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

df = df.replace({'gender': mapping})
before_mapping = df.copy()

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
             "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]


data = df[data_feat]
info = df[info_feat]
scaler = preprocessing.StandardScaler().fit(data)

scaled = scaler.transform(data)

colours = np.array(color_palette('pastel').as_hex())
labels = info['shock'].to_numpy().astype(int)

# %%
# Plotting
# --------

fig, axes = plt.subplots(4, 4, figsize=(15, 15))
fig.tight_layout()
axes=axes.flatten()
for ax in axes:
    sns.despine(ax=ax, left=False)
    ax.get_xaxis().set_ticks([])
for i in [3,12,15]:
    axes[i].axis('off')

tsne_points = pickle.load(open(TSNE_POINTS_PATH, 'rb'))

clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(tsne_points)
outliers = -1 in clustering.labels_
clusters = clustering.labels_

before_mapping['cluster'] = clusters

if outliers:
    before_mapping = before_mapping.loc[before_mapping['cluster'] != -1]


N_CLUSTERS = len(set(clusters))
colours = colours[:N_CLUSTERS]


if outliers:
    labels = ["Outliers"] + [f"Cluster {i}" for i in range(N_CLUSTERS - 1)]
else:
    labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

i = 0
for feat in sorted(data_feat):
    g = sns.boxplot(ax=axes[i], x="cluster", y=feat,
                     data=before_mapping, palette="pastel")
    axes[i].get_xaxis().set_ticks([])
    axes[i].set(xlabel=label_mapping[feat], ylabel=None)

    if i == 2:
        c1 = mpatches.Patch(color=colours[0], label='Cluster 1')
        c2 = mpatches.Patch(color=colours[1], label='Cluster 2')
        axes[3].legend(handles=[c1, c2], loc='center')
        i += 1
    i += 1

for feat in sorted(info_feat):
    if feat != 'gender':
        sns.barplot(ax=axes[i], x="cluster", y=feat,
                         data=before_mapping, palette="pastel")
    else:
        sns.barplot(ax=axes[i], x="cluster", y=feat,
                         data=before_mapping, palette="muted", estimator=lambda x: 1, ci=None)
        g = sns.barplot(ax=axes[i], x="cluster", y=feat,
                         data=before_mapping, palette="pastel")
        g.text(0, 0.25, 'Male', color='black', ha="center")
        g.text(0, 0.75, 'Female', color='black', ha="center")
        g.text(1, 0.25, 'Male', color='black', ha="center")
        g.text(1, 0.75, 'Female', color='black', ha="center")

    axes[i].get_xaxis().set_ticks([])
    axes[i].set(xlabel=label_mapping[feat], ylabel=None)

    i += 1 if i != 11 else 2

# fig.savefig("tsne_cluster_stats.pdf", bbox_inches='tight')
plt.show()


# %%
# t-SNE DBSCAN Clustering
# -----------------------

colours = np.array(sns.color_palette("pastel", as_cmap=True))
colours = np.insert(colours, 0, "#737373")
colours = dict(zip(list(range(-1, len(colours) - 1)), colours))

plt.figure()
ind_list = np.random.choice(len(df.index), 5000)
tsne_points_sampled = np.take(tsne_points, ind_list, axis=0)
clusters = np.take(clusters, ind_list, axis=0)
scatter = sns.scatterplot(x=tsne_points_sampled[:, 0], y=tsne_points_sampled[:, 1],
                          hue=clusters, palette=colours, linewidth=0,
                          s=10)
handles, _  =  scatter.get_legend_handles_labels()

scatter.legend(handles, labels, loc='lower right', borderpad=0.2,labelspacing=0.2)
# plt.savefig("tsne_dbscan.pdf", bbox_inches='tight')
plt.show()


# %%
# t-SNE Sheppard Diagram
# ----------------------

_, fig = distance_metrics(scaled, tsne_points, 6000, '', verbose=False)
# fig.savefig("tsne_sheppard.png", bbox_inches='tight', dpi=300)
plt.show()
