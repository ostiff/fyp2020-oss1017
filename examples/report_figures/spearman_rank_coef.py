"""
Spearman's Rank Coefficient
===========================

"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'font.size': 18,
    'pgf.rcfonts': False
})

x = np.array(np.arange(-1.9,2,0.1))
y_pos_corr = np.tan(x/1.7)
y_neg_corr = np.tan(-x/1.7)
y_no_corr = np.random.uniform(low=-1.9, high=1.9, size=len(x))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes:
    ax.set(aspect='equal')
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

# 1 corr
sns.scatterplot(ax=axes[0], x=x, y=y_pos_corr, color=".3")
axes[0].set_title(r'$\rho = 1$')

# 0 corr
sns.scatterplot(ax=axes[1], x=x, y=y_no_corr, color=".3")
axes[1].set_title(r'$\rho \approx 0$')

# -1 corr
sns.scatterplot(ax=axes[2], x=x, y=y_neg_corr, color=".3")
axes[2].set_title(r'$\rho = -1$')

fig.savefig("spearman_corr.pdf", bbox_inches='tight')
