"""
PCA for dimensionality reduction (dengue)
================================================

Training attributes: `age`, `weight`, `plt`, `haematocrit_percent`,
`body_temperature`.


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from pkgname.core.AE.autoencoder import  set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule


SEED = 0

# Set seed
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
df = df.replace({'gender': mapping})

info_feat = ["dsource", "shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]


train, test = train_test_split(df, test_size=0.2, random_state=SEED)

train_data = train[data_feat]
test_data = test[data_feat]
train_info = train[info_feat]
test_info = test[info_feat]

scaler = preprocessing.StandardScaler().fit(train_data)

train_scaled = scaler.transform(train_data.to_numpy())
test_scaled = scaler.transform(test_data.to_numpy())

pca = PCA(n_components=2)
X = pca.fit(train_scaled).transform(train_scaled)
y = train_info['shock'].to_numpy()

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise']
lw = 2

for color, i, target_name in zip(colors, [0, 1], ['No shock', 'shock']):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Dengue dataset')
plt.show()