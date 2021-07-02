"""
Autoencoder experiment figures
==============================

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

from sklearn import preprocessing, mixture
from seaborn import color_palette
from sklearn.model_selection import train_test_split
from tableone import TableOne
from torch.utils.data import DataLoader

from core.evaluation.dr_evaluation import distance_metrics
from pkgname.utils.data_loader import load_dengue, IQR_rule
from definitions import ROOT_DIR
from sklearn.cluster import DBSCAN, KMeans

from utils.plot_utils import plot_results, formatTable

sys.path.insert(0, os.path.abspath('.'))

# mpl.use("pgf")
mpl.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    # 'font.family': 'serif',
    'font.weight': 'light',
    'font.size': 16,
    #
    #
    # 'text.usetex': True,
    # 'font.size': 18,
})

SEED = 0
MODEL_PATH = os.path.join(ROOT_DIR, 'examples', 'report_figures', 'ae_figures', 'ae_sig_3')
np.random.seed(SEED)


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

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
             "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

data = df[data_feat]
info = df[info_feat]
scaled = preprocessing.MinMaxScaler().fit_transform(data)
batch_size=32
loader = DataLoader(scaled, batch_size, shuffle=False)
model = pickle.load(open(MODEL_PATH, 'rb'))
ae_points = model.encode_inputs(loader)

colours = np.array(color_palette('pastel').as_hex())

# %%
# AE
# -------

# OVERALL CLUSTER COMP.
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
fig.tight_layout()
axes=axes.flatten()
for ax in axes:
    sns.despine(ax=ax, left=False)
    ax.get_xaxis().set_ticks([])
for i in [3,12,15]:
    axes[i].axis('off')


gmm = mixture.GaussianMixture(n_components=3,
                                        covariance_type='full', random_state=SEED).fit(ae_points)
clusters = gmm.predict(ae_points)
clusters_k_means = KMeans(n_clusters=3, random_state=SEED).fit_predict(ae_points)


before_mapping['cluster'] = clusters

# %%
# Plotting
# --------

N_CLUSTERS = len(set(clusters))
colours = colours[:N_CLUSTERS]


labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

i = 0
for feat in sorted(data_feat):
    g = sns.boxplot(ax=axes[i], x="cluster", y=feat,
                     data=before_mapping, palette="pastel")
    axes[i].get_xaxis().set_ticks([])
    axes[i].set(xlabel=label_mapping[feat], ylabel=None)

    if i == 2:
        c1 = mpatches.Patch(color=colours[0], label='Cluster 0')
        c2 = mpatches.Patch(color=colours[1], label='Cluster 1')
        c3 = mpatches.Patch(color=colours[2], label='Cluster 2')

        axes[3].legend(handles=[c1, c2, c3], loc='center')
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
        g.text(2, 0.25, 'Male', color='black', ha="center")
        g.text(2, 0.75, 'Female', color='black', ha="center")

    axes[i].get_xaxis().set_ticks([])
    axes[i].set(xlabel=label_mapping[feat], ylabel=None)

    i += 1 if i != 11 else 2

# fig.savefig("ae_cluster_stats.pdf", bbox_inches='tight')
plt.show()

# %%
# Intra-cluster comparison
# ------------------------
fig, axes = plt.subplots(1, 6, figsize=(15, 4))
fig.tight_layout()
axes=axes.flatten()
for ax in axes:
    sns.despine(ax=ax, left=False)
    ax.get_xaxis().set_ticks([])


for i, feat in enumerate(sorted(data_feat)):
    g = sns.boxplot(ax=axes[i], x="cluster", y=feat,
                     data=before_mapping[before_mapping['cluster'] == 1], hue='shock', palette="Set3")
    axes[i].get_xaxis().set_ticks([])
    axes[i].set(xlabel=label_mapping[feat], ylabel=None)
    g.legend_.remove()

axes[-1].axis('off')
legend_colours = np.array(color_palette('Set3').as_hex())

c1 = mpatches.Patch(color=legend_colours[0], label='False')
c2 = mpatches.Patch(color=legend_colours[1], label='True')
axes[-1].legend(handles=[c1, c2], loc='center', title='Shock')

# fig.savefig("ae_cluster0_shock.pdf", bbox_inches='tight')
plt.show()



# %%
# GMM clustering
# --------------
colours = np.array(sns.color_palette("pastel", as_cmap=True))
# colours = np.insert(colours, 0, "#737373")
colours = dict(zip(list(range(len(colours))), colours))

plt.figure()
ind_list = np.random.choice(len(df.index), 5000)
ae_points_subset = np.take(ae_points, ind_list, axis=0)
clusters = np.take(clusters, ind_list, axis=0)

plot_results(ae_points_subset, clusters, gmm.means_, gmm.covariances_, colours, labels)
# plt.savefig("ae_gmm.pdf", bbox_inches='tight')
plt.show()

# %%
# k-means clustering
# ------------------

clusters_k_means = np.take(clusters_k_means, ind_list, axis=0)
plt.figure()
splot = sns.scatterplot(x=ae_points_subset[:, 0], y=ae_points_subset[:, 1], hue=clusters_k_means, palette=colours,
                            linewidth=0, s=10)
handles, _ = splot.get_legend_handles_labels()
splot.legend(handles, labels, loc='lower right', borderpad=0.2, labelspacing=0.2)
# plt.savefig("ae_kmeans.pdf", bbox_inches='tight')
plt.show()


# %%
# AE Sheppard Diagram
# -------------------
_, fig = distance_metrics(scaled, ae_points, 6000, '', verbose=False)
# fig.savefig("ae_sheppard.png", bbox_inches='tight', dpi=300)
plt.show()


# %%
# Shock in latent dimension
# -------------------------
shock = np.take(before_mapping['shock'].to_numpy(), ind_list, axis=0)
plt.figure()
splot = sns.scatterplot(x=ae_points_subset[:, 0], y=ae_points_subset[:, 1], hue=shock, palette='pastel',
                            linewidth=0, s=10)
legend_colours = np.array(color_palette('pastel').as_hex())

plt.legend(title='Shock', loc='lower right',borderpad=0.2,labelspacing=0.2)
# plt.savefig("ae_sig_shock.pdf", bbox_inches='tight')
plt.show()
