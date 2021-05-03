"""
t-SNE: Pathology dataset 1
===========================

Training attributes: FBC panel

Attributes used in cluster comparison: `age`, `gender`, `covid_confirmed`.

"""
# Libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tableone import TableOne

from pkgname.utils.data_loader import load_pathology
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger

logger = Logger('TSNE_Pathology', compress=False)

SEED = 0
TSNE_n_components = 2
TSNE_perplexity = 30
TSNE_early_exaggeration = 100
TSNE_learning_rate = 200
DBSCAN_eps = 5
DBSCAN_min_samples = 50

np.random.seed(SEED)

# %%
# Dataset
# --------
#
# Load dengue dataset. Perform forward and backwards fill after grouping by patient.
# Does not make use of the `d001` dataset because it does not contain: `abdominal_pain`,
# `bleeding_mucosal`, `bleeding_skin`, `body_temperature`.
# To reduce computation time aggregate patient data to only have one tuple per patient.


df = load_pathology(usedefault=True, dropna=True)
del df["_uid"]
del df["dateResult"]


# %%
# t-SNE
# --------
#
# Use t-SNE on the z-score scaled data.

info_feat = ["GenderID", "patient_age", "covid_confirmed"]
data_feat = ["EOS", "MONO", "BASO", "NEUT",
             "RBC", "WBC", "MCHC", "MCV",
             "LY", "HCT", "RDW", "HGB",
             "MCH", "PLT", "MPV", "NRBCA"]

info = df[info_feat]
data = df[data_feat]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(data.values)

X_embedded = TSNE(n_components=TSNE_n_components,
                  perplexity=TSNE_perplexity,
                  early_exaggeration=TSNE_early_exaggeration,
                  learning_rate=TSNE_learning_rate,
                  random_state=SEED, n_jobs=-1).fit_transform(x)

logger.save_object(X_embedded, "X_embedded")


plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=info['covid_confirmed'])
logger.add_plt(plt.gcf())
plt.show()

# %%
# DBSCAN
# --------
#
# Identify clusters using DBSCAN

DBSCAN_eps = 5
DBSCAN_min_samples = 5

clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(X_embedded)
outliers = -1 in clustering.labels_
clusters = [x+1 for x in clustering.labels_] if outliers else clustering.labels_

# %%
# Plotting
# --------

N_CLUSTERS = len(set(clusters))

colours = colours[:N_CLUSTERS]

scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=clusters, cmap=ListedColormap(colours))

if outliers:
    labels = ["Outliers"] + [f"Cluster {i}" for i in range(N_CLUSTERS-1)]
else:
    labels= [f"Cluster {i}" for i in range(N_CLUSTERS)]

plt.legend(handles=scatter.legend_elements()[0], labels=labels)
plt.title('t-SNE + DBSCAN')
logger.add_plt(plt.gcf())
plt.show()


# %%
# Cluster analysis
# ----------------
#
# Table

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    df['cluster'] = clusters

columns = info_feat+data_feat
nonnormal = list(df[columns].select_dtypes(include='number').columns)
nonnormal.remove('GenderID')
categorical = list(set(columns).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)

table = TableOne(df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', missing=False)

html = formatTable(table, colours, labels)
logger.append_html(html.render())
html


# %%
# These attributes were not used to train the model.

fig, html = plotBox(data=info,
                    features=info_feat,
                    clusters=clusters,
                    colours=colours,
                    labels=labels,
                    title="Attributes not used in training",
                    )
logger.append_html(html)
fig

#%%
# The following attributes were used to train the model.

fig, html = plotBox(data=data,
                    features=data_feat,
                    clusters=clusters,
                    colours=colours,
                    labels=labels,
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
        'TSNE_n_components': TSNE_n_components,
        'TSNE_perplexity': TSNE_perplexity,
        'TSNE_early_exaggeration': TSNE_early_exaggeration,
        'TSNE_learning_rate': TSNE_learning_rate,
        'DBSCAN_eps': DBSCAN_eps,
        'DBSCAN_min_samples': DBSCAN_min_samples,
        'info_feat': info_feat,
        'data_feat': data_feat
    }
)

logger.create_report()
