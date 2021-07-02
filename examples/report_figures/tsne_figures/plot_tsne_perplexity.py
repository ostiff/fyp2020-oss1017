"""
Effect of perplexity on t-SNE results
=====================================

"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from seaborn import color_palette

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pkgname.utils.data_loader import load_dengue, IQR_rule
from definitions import ROOT_DIR

sys.path.insert(0, os.path.abspath('.'))

# mpl.use("pgf")
mpl.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,
    'font.size': 18,
})

N_points = 2000
SEED = 0
np.random.seed(SEED)

TSNE_POINTS_PATH = os.path.join(ROOT_DIR, 'examples', 'report_figures', 'tsne_figures')


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

# Points to sample
ind_list = np.random.choice(len(df.index), N_points)
df = df.iloc[ind_list]

info = df[info_feat]
data = df[data_feat]

scaler = preprocessing.StandardScaler().fit(data)



data = df[data_feat]
info = df[info_feat]

scaled = scaler.transform(data)

# colours = np.array(color_palette('viridis', 2).as_hex())
labels = info['shock'].to_numpy().astype(int)

# %%
# t-SNE
# -------
paths = [(os.path.join(TSNE_POINTS_PATH, 'perp_5'), '5'),
         (os.path.join(TSNE_POINTS_PATH, 'perp_40'), '40'),
         (os.path.join(TSNE_POINTS_PATH, 'perp_200'), '200'),]


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes:
    ax.set(aspect='equal')
    ax.set(xlim=(-120, 120), ylim=(-120, 120))


for i, (path, perp) in enumerate(paths):
    colours = np.array(color_palette('pastel').as_hex())
    colours = dict(zip(list(range(len(colours))), colours))

    tsne_points = np.take(pickle.load(open(path, 'rb')), ind_list, axis=0)

    sns.scatterplot(ax=axes[i], x=tsne_points[:,0], y=tsne_points[:,1],
                    hue=info['shock'], palette=colours, linewidth=0, s=15, legend=(i==2))
    axes[i].set_title(f'Perplexity: {perp}')

plt.legend(title='Shock', loc='upper right',borderpad=0.2,labelspacing=0.2)
# fig.savefig("tsne_diff_perp.pdf", bbox_inches='tight')
plt.show()
