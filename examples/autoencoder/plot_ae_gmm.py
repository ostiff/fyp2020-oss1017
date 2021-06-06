"""
Auto-encoder w/ GMM clustering
==============================

"""

import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from pkgname.core.AE.autoencoder import get_device, set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule
from definitions import ROOT_DIR
from tableone import TableOne
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture

# --------------
# Load data
# --------------

SEED = 0
batch_size = 16
MODEL_PATH = os.path.join(ROOT_DIR, 'examples', 'autoencoder', 'model')
N_CLUSTERS = 3
# Set seed
set_seed(SEED)

# Get device
device = get_device(False)

features = ["dsource", "date", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]
info_feat = ["dsource", "shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
             "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

before_fill = load_dengue(usecols=['study_no'] + features)
before_fill = before_fill.loc[before_fill['age'] <= 18]
before_fill = IQR_rule(before_fill, ['plt', 'haematocrit_percent', 'body_temperature'])

df = before_fill.copy()

before_fill = before_fill.dropna(subset=data_feat + ['date'])

for feat in features:
    df[feat] = before_fill.groupby('study_no')[feat].ffill().bfill()

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

mapping = {'Female': 0, 'Male': 1}
df = df.replace({'gender': mapping})

train, test = train_test_split(df, test_size=0.2, random_state=SEED)

train_data = train[data_feat]
test_data = test[data_feat]
train_info = train[info_feat]
test_info = test[info_feat]

scaler = preprocessing.MinMaxScaler().fit(train_data)

train_scaled = scaler.transform(train_data.to_numpy())
test_scaled = scaler.transform(test_data.to_numpy())

loader_train = DataLoader(train_scaled, batch_size, shuffle=False)
loader_test = DataLoader(test_scaled, batch_size, shuffle=False)

model = pickle.load(open(MODEL_PATH, 'rb'))
encoded_train = model.encode_inputs(loader_train)

plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=train_info['shock'])
plt.title('AE shock in latent space (testing data)')
plt.show()



color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-3., 1.)
    plt.ylim(-3.25, 0.5)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=N_CLUSTERS, covariance_type='full').fit(encoded_train)
plot_results(encoded_train, gmm.predict(encoded_train), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')



# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=N_CLUSTERS,
                                        covariance_type='full').fit(encoded_train)
plot_results(encoded_train, dpgmm.predict(encoded_train), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()


# Table
labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

mapping = {0: 'Female', 1: 'Male'}
table_df = train.replace({'gender': mapping})

table_df['cluster'] = dpgmm.predict(encoded_train)

columns = info_feat+data_feat
nonnormal = list(table_df[columns].select_dtypes(include='number').columns)
categorical = list(set(columns).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)

rename = {'haematocrit_percent': 'hct',
          'body_temperature': 'temperature'}

table = TableOne(table_df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', rename=rename, missing=False)


print(table.tabulate(tablefmt="fancy_grid"))
