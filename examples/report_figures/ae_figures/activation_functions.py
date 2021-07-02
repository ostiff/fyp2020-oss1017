"""
Activation functions
====================

"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,
    'font.size': 18,
    'pgf.rcfonts': False
})


def relu(x):
    """Returns a ReLU."""
    return np.maximum(x, 0)


def sigmoid(x):
    """Returns a Sigmoid."""
    return 1 / (1 + np.exp(-x))



points = 300
x = np.linspace(-4, 4, num=points)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes = axes.flatten()
for ax in axes:
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

axes[0].set(xlim=(-1.3, 1.3), ylim=(-0.1, 1.3))
axes[1].set(xlim=(-4, 4), ylim=(-0.1, 1.1))

sns.set_theme()


sns.lineplot(ax=axes[0], x=x, y=relu(x), color=".3")
axes[0].set_title(r'ReLU')

sns.lineplot(ax=axes[1], x=x, y=sigmoid(x), color=".3")
axes[1].set_title(r'Sigmoid')


# fig.savefig("ae_activation_functions.pdf", bbox_inches='tight')
plt.show()
