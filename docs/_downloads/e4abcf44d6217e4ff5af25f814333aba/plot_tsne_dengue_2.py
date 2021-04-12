"""
t-SNE: Dengue dataset 2
============================

Training attributes: `age`, `gender`, `weight`, `plt`, `haematocrit_percent`,
  `body_temperature`.

Attributes used in cluster comparison: `bleeding`, `shock`, `bleeding_gum`,
 `abdominal_pain`, `ascites`, `bleeding_mucosal`, `bleeding_skin`.

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

mapping = {'Female': 1, 'Male': 2}
df = df.replace({'gender': mapping})

features = ["date", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

for feat in features:
    df[feat] = df.groupby('study_no')[feat].ffill().bfill()

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
print("len", len(df.index))
# %%
# t-SNE
# --------
#
# Use t-SNE on the z-score scaled data.

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin", ]
info_df = df[info_feat]

data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]
data_df = df[data_feat]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(data_df.values)

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


info_df['cluster'] = clustering.labels_

_, ax1 = plt.subplots(len(info_feat), 1, figsize=(5, 5 * len(info_feat)))

for i, feat in enumerate(info_feat):
    info_df.boxplot(feat,'cluster', ax=ax1[i], showmeans=True)

plt.show()



data_df['cluster'] = clustering.labels_

_, ax1 = plt.subplots(len(data_feat), 1, figsize=(5, 5 * len(data_feat)))

for i, feat in enumerate(data_feat):
    data_df.boxplot(feat,'cluster', ax=ax1[i], showmeans=True)

plt.show()
