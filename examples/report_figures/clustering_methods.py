"""
Clustering method comparison
============================

"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,
})

SEED = 1
np.random.seed(SEED)

# %%
# Generate datasets
# -----------------
n_samples = 1100

noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.06)
blobs = datasets.make_blobs(n_samples=n_samples,
                            cluster_std=[1.0, 2.8, 0.5],
                            random_state=SEED)

datasets = [
    (noisy_moons, {'eps': .3, 'n_clusters': 2}),
    (blobs, {'eps': .19, 'n_clusters': 3})]


# %%
# Create figure
# -------------

fig, axes = plt.subplots(2, 3, figsize=(3 * 2 + 3, 6))
axes = axes.flatten()
for ax in axes:
    ax.set(aspect='equal')
    ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

plot_num = 0

for i_dataset, (dataset, params) in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    clustering_algorithms = (
        (r'k-Means', cluster.KMeans(n_clusters=params['n_clusters'])),
        (r'DBSCAN', cluster.DBSCAN(eps=params['eps'])),
        (r'GMM', mixture.GaussianMixture(n_components=params['n_clusters']))
    )

    for name, algorithm in clustering_algorithms:

        algorithm.fit(X)

        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        if i_dataset == 0:
            axes[plot_num].set_title(name)

        colors = np.array(sns.color_palette("pastel", as_cmap=True))
        colors = np.insert(colors, 0, "#737373")
        colors = dict(zip(list(range(-1,len(colors)-1)), colors))

        sns.scatterplot(ax=axes[plot_num], x=X[:, 0], y=X[:, 1], size=4,
                        hue=y_pred, palette=colors, legend=False)

        plot_num += 1

fig.tight_layout(h_pad=1, w_pad=2)
# fig.savefig("clustering_methods.pdf", bbox_inches='tight')
plt.show()
