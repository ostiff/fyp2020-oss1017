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
import matplotlib.gridspec as gridspec
from minisom import MiniSom
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tableone import TableOne

# Utils
from pkgname.utils.som_utils import diff_graph_hex, feature_maps, project_hex
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger
from pkgname.core.AE.autoencoder import set_seed
from pkgname.utils.print_utils import suppress_stderr, suppress_stdout
import pkgname.core.evaluation.dr_evaluation as dr_evaluation

import matplotlib as mpl
# mpl.use("pgf")
mpl.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,
    # 'font.size': 18,
})

logger = Logger('SOM_Dengue', enable=False)

# Configuration
rcParams.update({'figure.autolayout': True})

N_CLUSTERS = 3
SOM_X_SIZE = 55
SOM_Y_SIZE = 13
SOM_SIGMA = 10
SOM_lr = 0.25
SOM_ACTIVATION_DIST = 'euclidean'
SOM_NEIGHBOURHOOD = 'bubble'
SEED = 0
np.random.seed(SEED)
set_seed(SEED)

# %-----------
# Load dataset
# ------------

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
before_mapping = df.copy()

df = df.replace({'gender': mapping})

info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin", "gender"]
data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

info = df[info_feat]
data = df[data_feat]
print('len', len(data.index))
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(data.values)

# ----------------------
# Train SOM
# ----------------------
# Create SOM

with suppress_stderr() and suppress_stdout():
    som = MiniSom(SOM_X_SIZE, SOM_Y_SIZE, x.shape[1],
        topology='hexagonal',
        activation_distance=SOM_ACTIVATION_DIST,
        neighborhood_function=SOM_NEIGHBOURHOOD,
        sigma=SOM_SIGMA, learning_rate=SOM_lr,
        random_seed=SEED)

    # Train
    som.pca_weights_init(x)
    som.train_random(x, 1000000, verbose=True)

projected_points = project_hex(som, x)

logger.save_object(projected_points, "som_embedded")

diff_graph_hex(som, show=False, printout=False)
plt.savefig('som_diff_graph.pdf', bbox_inches='tight')
logger.add_plt(plt.gcf())
plt.show()

feature_maps(som, feature_names=data_feat, cols=2, show=False, printout=False)
logger.add_plt(plt.gcf())
plt.show()

# Evaluate dimensionality reduction
# Distance metrics
res, fig = dr_evaluation.distance_metrics(x, projected_points, n_points=1000, method_name='SOM', verbose=True)
logger.add_plt(plt.gcf(), ext='png')
plt.show()
logger.add_parameters(res)

# Density metrics
res, fig = dr_evaluation.density_metrics(info, projected_points, ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
           "bleeding_mucosal", "bleeding_skin"], 'SOM')
logger.add_plt(plt.gcf(), ext='png')
plt.show()
logger.add_parameters(res)


# ------------------------
# Shock label distribution
# ------------------------
label_names = {False:'False', True:'True',0:'False', 1:'True'}
target = info['shock'].to_numpy()
labels_map = som.labels_map(x, [label_names[t] for t in target])

fig = plt.figure(figsize=(15, 4))
the_grid = gridspec.GridSpec(SOM_Y_SIZE, SOM_X_SIZE, fig)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names.values()]
    plt.subplot(the_grid[SOM_Y_SIZE-1-position[1],
                         position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
plt.subplots_adjust(wspace=0, hspace=0)

plt.legend(patches, label_names.values(), bbox_to_anchor=(-25,-2), title="Shock")
# plt.savefig('som_shock_pies.pdf', dpi=300, bbox_inches='tight')
plt.show()


# ----------
# Clustering
# ----------

colours = colours[:N_CLUSTERS]

proj = project_hex(som, x)

# Perform clustering on encoded inputs
cluster = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(proj)

labels = [f"Cluster {i}" for i in range(1,N_CLUSTERS+1)]

scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster, cmap=ListedColormap(colours))
plt.legend(handles=scatter.legend_elements()[0], labels=labels, loc="upper right")
plt.savefig('som_k_means.pdf', bbox_inches='tight')
logger.add_plt(plt.gcf())
plt.show()

# %%
#

# ----------------
# Cluster analysis
# ----------------
#
# Table
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    before_mapping['cluster'] = cluster

columns = info_feat+data_feat
nonnormal = list(before_mapping[columns].select_dtypes(include='number').columns)
categorical = list(set(columns).difference(set(nonnormal))) + ['dsource']
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
                    clusters=cluster,
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
                    clusters=cluster,
                    colours=colours,
                    labels=labels,
                    title="Attributes used in training",
                    )
logger.append_html(html)
fig


logger.save_parameters(
    {
        'SEED': SEED,
        'features': features,
        'info_feat': info_feat,
        'data_feat': data_feat,
        'SOM_X_SIZE': SOM_X_SIZE,
        'SOM_Y_SIZE': SOM_Y_SIZE,
        'SOM_SIGMA': SOM_SIGMA,
        'SOM_lr': SOM_lr,
        'SOM_ACTIVATION_DIST': SOM_ACTIVATION_DIST,
        'SOM_NEIGHBOURHOOD': SOM_NEIGHBOURHOOD
    }
)

logger.create_report()