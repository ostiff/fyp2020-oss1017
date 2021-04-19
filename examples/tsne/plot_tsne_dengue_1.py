"""
t-SNE: Dengue dataset 1
=======================

Training attributes: `bleeding`, `plt`, `shock`, `haematocrit_percent`,
`bleeding_gum`, `abdominal_pain`, `ascites`, `bleeding_mucosal`,
`bleeding_skin`, `body_temperature`.

Attributes used in cluster comparison: `age`, `gender`, `weight`.

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

from pkgname.utils.data_loader import load_dengue
from pkgname.utils.plot_utils import plotBox, formatTable
from pkgname.utils.log_utils import Logger

logger = Logger('TSNE_Dengue')

SEED = 0
np.random.seed(SEED)

# %%
# Dataset
# --------
#
# Load dengue dataset. Perform forward and backwards fill after grouping by patient.
# Does not make use of the `d001` dataset because it does not contain: `abdominal_pain`,
# `bleeding_mucosal`, `bleeding_skin`, `body_temperature`.
# To reduce computation time aggregate patient data to only have one tuple per patient.


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

before_mapping = df
mapping = {'Female': 0, 'Male': 1}
df = df.replace({'gender': mapping})

# %%
# t-SNE
# --------
#
# Use t-SNE on the z-score scaled data.

info_feat = ["dsource", "age", "gender", "weight"]
data_feat = ["bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

info = df[info_feat]
data = df[data_feat]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(data.values)

TSNE_n_components = 2
TSNE_perplexity = 500

X_embedded = TSNE(n_components=TSNE_n_components, perplexity=TSNE_perplexity,
                  random_state=SEED, n_jobs=-1).fit_transform(x)

logger.save_object(X_embedded, "X_embedded")


# %%
# DBSCAN
# --------
#
# Identify clusters using DBSCAN

DBSCAN_eps = 10
DBSCAN_min_samples = 5

clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(X_embedded)
outliers = -1 in clustering.labels_
clusters = [x+1 for x in clustering.labels_] if outliers else clustering.labels_

# %%
# Plotting
# --------

N_CLUSTERS = len(set(clusters))

colours = ["red", "blue", "limegreen", "orangered", "yellow",
           "violet", "salmon", "slategrey", "green", "crimson"][:N_CLUSTERS]

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

    before_mapping['cluster'] = clusters

columns = info_feat+data_feat
nonnormal = list(before_mapping[columns].select_dtypes(include='number').columns)
categorical = list(set(columns).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)

rename = {'haematocrit_percent': 'hct',
          'body_temperature': 'temperature'}

table = TableOne(before_mapping, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', rename=rename, missing=False)

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
                    #path="a.html"
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
                    #path="b.html"
                    )
logger.append_html(html)
fig


# Log parameters
logger.save_parameters(
    {
        'SEED': SEED,
        'TSNE_n_components': TSNE_n_components,
        'TSNE_perplexity': TSNE_perplexity,
        'DBSCAN_eps': DBSCAN_eps,
        'DBSCAN_min_samples': DBSCAN_min_samples,
        'features': features,
        'info_feat': info_feat,
        'data_feat': data_feat
    }
)

logger.create_report()
