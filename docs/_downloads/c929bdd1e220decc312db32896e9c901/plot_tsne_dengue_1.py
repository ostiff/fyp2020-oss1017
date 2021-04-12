"""
t-SNE: Dengue dataset 1
============================

Training attributes: `bleeding`, `plt`, `shock`, `haematocrit_percent`,
 `bleeding_gum`, `abdominal_pain`, `ascites`, `bleeding_mucosal`,
  `bleeding_skin`, `body_temperature`.

Attributes used in cluster comparison: `age`, `gender`, `weight`.

"""
# Libraries
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import matplotlib.pyplot as plt

from pkgname.utils.data_loader import load_dengue
pd.set_option('display.max_columns', None)


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

df = load_dengue(usedefault=True)

mapping = {'Female': 1, 'Male': 2, True: 1, False: 0}
df = df.replace({'gender': mapping, 'bleeding': mapping, 'shock': mapping,
                 'bleeding_gum': mapping, 'abdominal_pain': mapping, 'ascites': mapping,
                 'bleeding_mucosal': mapping, 'bleeding_skin': mapping})

features = ["date", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

# for feat in features:
#     df[feat] = df.groupby('study_no')[feat].ffill().bfill()

df = df.loc[df['age'] <= 18]
#df = df.dropna()

df = df.groupby(by="study_no", dropna=False).agg(
    date=pd.NamedAgg(column="date", aggfunc="last"),
    age=pd.NamedAgg(column="age", aggfunc="max"),
    gender=pd.NamedAgg(column="gender", aggfunc="first"),
    weight=pd.NamedAgg(column="weight", aggfunc=np.mean),
    bleeding=pd.NamedAgg(column="bleeding", aggfunc="max"),
    plt=pd.NamedAgg(column="plt", aggfunc="max"),
    shock=pd.NamedAgg(column="shock", aggfunc="max"),
    haematocrit_percent=pd.NamedAgg(column="haematocrit_percent", aggfunc="max"),
    bleeding_gum=pd.NamedAgg(column="bleeding_gum", aggfunc="max"),
    abdominal_pain=pd.NamedAgg(column="abdominal_pain", aggfunc="max"),
    ascites=pd.NamedAgg(column="ascites", aggfunc="max"),
    bleeding_mucosal=pd.NamedAgg(column="bleeding_mucosal", aggfunc="max"),
    bleeding_skin=pd.NamedAgg(column="bleeding_skin", aggfunc="max"),
    body_temperature=pd.NamedAgg(column="body_temperature", aggfunc=np.mean),
).dropna()

# %%
# t-SNE
# --------
#
# Use t-SNE on the z-score scaled data.

info = df[["date", "age", "gender", "weight"]]
data = df[["bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(data.values)

X_embedded = TSNE(n_components=2, perplexity=500, random_state=SEED).fit_transform(x)

# %%
# DBSCAN
# --------
#
# Identify clusters using DBSCAN

clustering = DBSCAN(eps=10, min_samples=5).fit(X_embedded)

# %%
# Plotting
# --------

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=clustering.labels_)
plt.title('t-SNE + DBSCAN')
plt.show()

info['cluster'] = clustering.labels_

_, ax0 = plt.subplots(3, 1, figsize=(5, 15))
info.boxplot('age','cluster', ax=ax0[0], showmeans=True)
info.boxplot('gender','cluster', ax=ax0[1], showmeans=True)
info.boxplot('weight','cluster', ax=ax0[2], showmeans=True)
plt.show()


data['cluster'] = clustering.labels_

_, ax1 = plt.subplots(10, 1, figsize=(5, 50))

features = ["bleeding", "plt", "shock", "haematocrit_percent", "bleeding_gum",
            "abdominal_pain", "ascites", "bleeding_mucosal", "bleeding_skin",
            "body_temperature"]

for i, feat in enumerate(features):
    data.boxplot(feat,'cluster', ax=ax1[i], showmeans=True)

plt.show()
