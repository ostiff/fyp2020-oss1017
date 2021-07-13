"""
Outcomes (numbers)
==================

This script is useful to understand the distribution
over the 2D projected space of the different categorical
features and/or outcomes.

"""

# Library
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl

# Specific
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from tableone import TableOne

# Specific
from pkgname.core.AE.autoencoder import get_device, set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import  format_table_bootstrap
from definitions import ROOT_DIR

# Local (also configures mpl and pd)
import _utils

# ------------------------------
# Constants
# ------------------------------
# Load constants
features = _utils.features
others = _utils.others
mappings = _utils.mappings
aggregation = _utils.aggregation
cmaps = _utils.cmaps

aggregation['pcr_dengue_serotype'] = 'first'
aggregation['serology_interpretation'] = 'first'
aggregation['dengue_interpretation'] = 'first'

mappings['pcr_dengue_serotype'] = {
    'DENV-1,DENV-2': 'Mixed',
    'DENV-1,DENV-3': 'Mixed',
    'DENV-1,DENV-4': 'Mixed',
    'DENV-2,DENV-3': 'Mixed',
    'DENV-2,DENV-4': 'Mixed',
}

outcomes = []

# Ensure all outcomes are in aggregation
for c in outcomes:
    if c not in aggregation:
        print("Adding... %23s | max" % c)
        aggregation[c] = 'max'

# ------------------------------
# Load data
# ------------------------------
# Load data
data = _utils.load_data()

# Raw
raw = data.copy(deep=True)

# Filter data (age, iqr, ...)
# .. note: Although previously we were filtering only
#          for adults, there is an interesting patient
#          (01nva-003-2164) who is 19 years old.
#
# .. note: Platelets has a very clear outliers 50000
#          which influences the quartile selection
#          in the quartile range rule (IQR)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000]
data = data[data.dsource != '01nva'] # 01nva has no pcr
#data = data[data.ast < 1500]  # extreme outlier
#data = data[data.pcr_dengue_load < 1e10]

# Filter outliers
data = IQR_rule(data, [
    'plt',
])

# Needs all features for projection
data = data.dropna(how='any', subset=features)

# Convert dtypes
data = data.convert_dtypes()

# Rename
# .. note: Done after convert_dtypes so that
#          it remains 0 or 1, otherwise it will
#          be transformed to boolean feature.
#          (its ok too).
data = data.replace(mappings)

# Show dtypes
print("\nDtypes:")
print(data.dtypes)

# ------------------------------
# Aggregate to worse combo
# ------------------------------
# Get worst state for patient
data_w = data.copy(deep=True) \
    .groupby(by="study_no", dropna=False) \
    .agg(aggregation) \
    .dropna(how='any', subset=features)

# Get full data
data_f = data.copy(deep=True) \
    .dropna(how='any', subset=features)

# Show data
print("\nData Full:")
print(data_f)
print("\nData Agg:")
print(data_w)
print(data_w[outcomes].sum())
print("\nData 01NVA:")
print(data_w[data_w.dsource == '01nva'])


# ------------------------------
# Load Model
# ------------------------------
# Load model
model = _utils.load_model()

# Show
print("\nModel:")
print(model)

# ------------------------------
# Projections
# ------------------------------
# .. note: The scaling method should have also
#          been saved when training the model.
# Scaler
scaler = MinMaxScaler().fit(data_f[features])

# Include encoded in aggregated
data_w[['x', 'y']] = model.encode_inputs( \
    DataLoader(scaler.transform(data_w[features]),
         16, shuffle=False))

# Include encoded in full (filtered)
data_f[['x', 'y']] = model.encode_inputs( \
    DataLoader(scaler.transform(data_f[features]),
         16, shuffle=False))

# Show
print("\nScaler:")
print(scaler)
print("\nData:")
print(data_w)


# ------------------------------
# Visualization
# ------------------------------
# Libraries
import matplotlib.pyplot as plt

# Specific
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Specific
from _utils import kde_mpl_plot
from _utils import kde_mpl_compute
from _utils import kde_mpl


def make_colormap(seq):
    """Return a LinearSegmentedColormap

    Parameters
    ----------
    seq: list
        A sequence of floats and RGB-tuples. The floats
        should be increasing and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def adjust_lightness(color, amount=0.5):
    """Adjusts the lightness of a color

    Parameters
    ----------
    color: string or vector
        The color in string, hex or rgb format.

    amount: float
        Lower values result in dark colors.
    """
    # Libraries
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], \
        max(0, min(1, amount * c[1])), c[2])


# -------------------------------------
# Plot aggregated categories
# -------------------------------------
from matplotlib.gridspec import GridSpec

# Copy data
aux = data_w.copy(deep=True)

# Outcomes to visualise
outcomes = [
    'pcr_dengue_serotype',
    'serology_interpretation',
    'dengue_interpretation',
    'dsource'
]

# For each outcome
for o in outcomes:
    # Compute n
    n = aux[o].nunique()

    # Colors
    c = mpl.cm.get_cmap('Set3')(np.linspace(0,1,n))

    # Create figure
    f1, axes1 = plt.subplots(2, n,
        figsize=(n*3.0, 2*2.5), sharex=True, sharey=True)

    # Display
    for i, (l,g) in enumerate(aux.groupby(o)):

        # Create colormap
        cmap = LinearSegmentedColormap .from_list("",
            ['white', adjust_lightness(c[i], 0.2)], 14)

        # Plot scatter
        axes1.flat[i].scatter(g.x, g.y,
            s=8, linewidth=0.5, edgecolor='k',
            label=l, color=c[i])
        axes1.flat[i].set(
            title='%s (%s)' % (l, g.shape[0]))

        # Plot kde
        kde_mpl(g.x, g.y, ax=axes1.flat[n+i],
                contour=True, cmap=cmap,
                xlim=(aux.x.min(), aux.x.max()),
                ylim=(aux.y.min(), aux.y.max()))

        # Adjust graphs due to missing colorbar
        divider = make_axes_locatable(axes1.flat[i])
        cax = divider.append_axes("right", size="25%", pad=.05)
        cax.remove()

    # Configure all axes
    for ax in axes1.flat:
        ax.legend(loc='lower left')
        ax.set(aspect='equal')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    # Configure
    plt.suptitle(o)
    plt.tight_layout()

# Show
plt.show()