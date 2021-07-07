"""
Utils
=====
"""

# Libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Specific
from scipy.stats import gaussian_kde


# ----------------------------
# set basic configuration
# ----------------------------
# Matplotlib options
mpl.rc('font', size=8)
mpl.rc('legend', fontsize=6)
mpl.rc('xtick', labelsize=6)
mpl.rc('ytick', labelsize=6)

# Set pandas configuration.
pd.set_option('display.max_colwidth', 14)
pd.set_option('display.width', 150)
pd.set_option('display.precision', 4)

# ------------------------------
# Constants
# ------------------------------
features = [
    "age",
    "weight",
    "plt",
    "haematocrit_percent",
    "body_temperature"
]

outcomes = [
    "severe",
    "warning",
    "mild"]

others = [
    'study_no',
    'dsource',
    'date'
]

mappings = {
    'gender': {
        'Female': 0,
        'Male': 1
    }
}

aggregation = {
    'dsource': 'last',
    'age': 'max',
    'weight': 'mean',
    'plt': 'min',
    'haematocrit_percent': 'max',
    'body_temperature': 'mean',
    'gender': 'first',
    'date': 'last',
    'shock': 'max'
}

cmaps = {
    'severe': 'Reds',
    'warning': 'Oranges',
    'mild': 'Blues'
}


def_data = 'resources/datasets/combined_tidy_v0.0.10.csv'
def_model = 'resources/models/ae_sig_3'

# ------------------------------
# Methods
# ------------------------------
def load_data(path=def_data):
    """Load dataset"""
    return pd.read_csv(path,
        parse_dates=['date'])

def load_model(path=def_model):
    """Load model"""
    return pickle.load(open(path, 'rb'))

def kde_mpl_compute(x, y, xlim=None, ylim=None, **kwargs):
    """Computes the gaussian kde.

    Parameters
    ----------

    Returns
    -------
    """
    try:
        # Plot density
        kde = gaussian_kde(np.vstack((x, y)), **kwargs)
    except Exception as e:
        print("Exception! %s" % e)
        return None, None, None

    # Parameters
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    # Set xlim and ylim
    if xlim is not None:
        xmin, xmax = xlim
    if ylim is not None:
        ymin, ymax = ylim

    #kde = stats.gaussian_kde(data)
    #xx, yy = np.mgrid[-3:3:.01, -1:4:.01]
    #density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)

    # evaluate on a regular grid
    xgrid = np.linspace(xmin, xmax, 100)
    ygrid = np.linspace(ymin, ymax, 100)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    zgrid = kde.evaluate(np.vstack([
        Xgrid.ravel(),
        Ygrid.ravel()
    ]))
    Zgrid = zgrid.reshape(Xgrid.shape)

    """
    from sklearn.neighbors import KernelDensity
    kde_skl = KernelDensity(bandwidth=0.05)
    kde_skl.fit(np.vstack([y, x]).T)
    Zgrid = np.exp(kde_skl.score_samples( \
        np.vstack([
            Ygrid.ravel(),
            Xgrid.ravel()
        ]).T)).reshape(Xgrid.shape)
    """
    # Return
    return xgrid, ygrid, Zgrid


def kde_mpl_plot(xgrid, ygrid, Zgrid, cmap='Reds',
                 ax=None, contour=True, cbar=True):
    """Plots the KDE

    .. note: The imshow function works in a different
             way and therefore we have to tranpose the
             image.

    Parameters
    ----------

    Returns
    -------
    """
    # Plot the result as an image
    ax.imshow(Zgrid.T,
              origin='lower', aspect='auto',
              extent=[min(xgrid),
                      max(xgrid),
                      min(ygrid),
                      max(ygrid)],
              cmap=cmap)

    # Plot contour
    if contour:
        ax.contour(xgrid, ygrid, Zgrid,
                   levels=14, linewidths=0.25, alpha=0.5,
                   linestyles='dashed', colors='k')
        cntr = ax.contourf(xgrid, ygrid, Zgrid,
                           levels=14, cmap=cmap)

        if cbar:
            cb = plt.colorbar(cntr, ax=ax)
        # cb.set_label('density')

    # Return
    return ax

def kde_mpl(x, y, xlim=None, ylim=None, cmap='Reds',
            ax=None, contour=True, cbar=True,
            **kwargs):
    """Plot gaussian kde matplotlib

    Parameters
    ----------
    x, y: arrays
        Numpy arrays with the 2D values.

    Returns
    -------

    """
    # Compute KDE
    xgrid, ygrid, Zgrid = \
        kde_mpl_compute(x, y, xlim, ylim, **kwargs)

    # Plot KDE
    if xgrid is not None:
        ax = kde_mpl_plot(xgrid, ygrid, Zgrid,
            cmap=cmap, ax=ax, contour=contour,
            cbar=cbar)
        return ax

    # Return
    return None

