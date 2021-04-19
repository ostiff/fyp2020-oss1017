"""
SOM: Dengue dataset
===================

..TODO: Add description

Training attributes: `age`, `weight`, `plt`, `haematocrit_percent`,
`body_temperature`.

Attributes used in cluster comparison: `bleeding`, `shock`, `bleeding_gum`,
`abdominal_pain`, `ascites`, `bleeding_mucosal`, `bleeding_skin`.

"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tableone import TableOne

# Utils
from pkgname.utils.som_utils import diff_graph_hex, feature_maps, project_hex
from pkgname.utils.data_loader import load_dengue
from pkgname.utils.plot_utils import plotBox, formatTable

# Configuration
rcParams.update({'figure.autolayout': True})

N_CLUSTERS = 3
SEED = 0
np.random.seed(SEED)

# %-----------
# Load dataset
# ------------

features = ["dsource", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

df = load_dengue(usecols=['study_no']+features)

for feat in features:
    df[feat] = df.groupby('study_no')[feat].ffill().bfill()

df = df.loc[df['age'] <= 18]
df = df.dropna()

df = df.groupby(by="study_no", dropna=False).agg(
    dsource=pd.NamedAgg(column="dsource", aggfunc="last"),
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
before_mapping = df

df = df.replace({'gender': mapping})

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

info = df[info_feat]
data = df[data_feat]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(data.values)

# ----------------------
# Train SOM
# ----------------------
# Create SOM
som = MiniSom(20, 20, x.shape[1],
    topology='hexagonal',
    activation_distance='euclidean',
    neighborhood_function='gaussian',
    sigma=3, learning_rate=0.05,
    random_seed=SEED)

# Train
som.pca_weights_init(x)
som.train_random(x, 10000000, verbose=True)

diff_graph_hex(som, show=True, printout=False)
feature_maps(som, feature_names=data_feat, cols=2, show=True, printout=False)

# ----------
# Clustering
# ----------

colours = ["red", "blue", "limegreen", "orangered", "yellow",
           "violet", "salmon", "slategrey", "green", "crimson"][:N_CLUSTERS]

proj = project_hex(som, x)

# Perform clustering on encoded inputs
cluster = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(proj)

labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster, cmap=ListedColormap(colours))
plt.legend(handles=scatter.legend_elements()[0], labels=labels)
plt.show()


# ----------------
# Cluster analysis
# ----------------
#
# Table
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    before_mapping['cluster'] = cluster

nonnormal = list(before_mapping[features].select_dtypes(include='number').columns)
categorical = list(set(features).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)

rename = {'haematocrit_percent': 'hct',
          'body_temperature': 'temperature'}

table = TableOne(before_mapping, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', rename=rename, missing=False)


print(table.tabulate(tablefmt="grid"))
# html = formatTable(table, colours, labels)
# html

