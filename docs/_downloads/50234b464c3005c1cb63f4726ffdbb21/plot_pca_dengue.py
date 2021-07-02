"""
PCA for dimensionality reduction (dengue)
================================================

Training attributes: `age`, `weight`, `plt`, `haematocrit_percent`,
`body_temperature`.


"""

import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tableone import TableOne

from pkgname.core.AE.autoencoder import set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger

import pkgname.core.evaluation.dr_evaluation as dr_evaluation

logger = Logger('PCA_Dengue', enable=False)
SEED = 0
N_CLUSTERS = 3

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

pca = PCA(n_components=2).fit(train_scaled)
encoded_test = pca.transform(test_scaled)
encoded_train = pca.transform(train_scaled)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))


# Evaluate dimensionality reduction
# Distance metrics
res, fig = dr_evaluation.distance_metrics(train_scaled, encoded_train, n_points=1000, method_name='PCA', verbose=True)
logger.add_plt(plt.gcf(), ext='png')
plt.show()
logger.add_parameters(res)

# Density metrics
res, fig = dr_evaluation.density_metrics(train_info, encoded_train, ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin"], 'PCA')
logger.add_plt(plt.gcf(), ext='png')
plt.show()
logger.add_parameters(res)


# ----------
# Clustering
# ----------

colours = colours[:N_CLUSTERS]


# %%
#


clusters_test = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(encoded_test)
clusters_train = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(encoded_train)

labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

scatter = plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=clusters_test, cmap=ListedColormap(colours))
plt.legend(handles=scatter.legend_elements()[0], labels=labels)
plt.title('k-means (testing data)')
logger.add_plt(plt.gcf())
plt.show()

# %%
#

scatter = plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=clusters_train, cmap=ListedColormap(colours))
plt.legend(handles=scatter.legend_elements()[0], labels=labels)
plt.title('k-means (training data)')
logger.add_plt(plt.gcf())
plt.show()

# %%
#


# Table
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    mapping = {0: 'Female', 1: 'Male'}
    table_df = test.replace({'gender': mapping})

    table_df['cluster'] = clusters_test

columns = info_feat+data_feat
nonnormal = list(table_df[columns].select_dtypes(include='number').columns)
categorical = list(set(columns).difference(set(nonnormal)))
columns = sorted(categorical) + sorted(nonnormal)

rename = {'haematocrit_percent': 'hct',
          'body_temperature': 'temperature'}

table = TableOne(table_df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                 groupby='cluster', rename=rename, missing=False)

html = formatTable(table, colours, labels)
logger.append_html(html.render())
html

# %%
#

fig, html = plotBox(data=train_info,
                    features=info_feat,
                    clusters=clusters_train,
                    colours=colours,
                    title="Attributes not used in training",
                    )
logger.append_html(html)
fig

# %%
#

fig, html = plotBox(data=train_data,
                    features=data_feat,
                    clusters=clusters_train,
                    colours=colours,
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
        'N_CLUSTERS': N_CLUSTERS,
        'features': features,
        'info_feat': info_feat,
        'data_feat': data_feat
    }
)

logger.create_report()