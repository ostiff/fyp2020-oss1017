"""
KDE - Outcomes
==============

This script is useful to explore the regions in which
each of the clinical outcomes (e.g. shock, leakage, ...)
are more common (higher density).

Notes:
    - The definition of the compound features can be changed.
      For instance, to the definition of severity defined by
      damien we have included more variables that are related.

    - There are some important filtering.
        age <= 18 => children
        plt => exxtreme outlier

    - The previously mentioned changes, affect also the
      contour maps displayed.

"""

# Library
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl

# Specific
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

# Specific
from pkgname.core.AE.autoencoder import get_device, set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule

# Local
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

# Define outcomes
outcomes = [
   "age"
]

# Ensure all outcomes are in aggregation
for c in outcomes:
    if c not in aggregation:
        print("Adding... %23s | max" % c)
        aggregation[c] = 'max'

nrows, ncols = 5, 8

# ------------------------------
# Load data
# ------------------------------
# Load data
data = _utils.load_data()

# Fill empty values (be careful!)
for c in outcomes:
    data[c] = data[c].fillna(0)

# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000]

# Filter outliers
data = IQR_rule(data, [
    'plt',
    #'haematocrit_percent',
    #'body_temperature'
])

# Rename
# .. note: Done after convert_dtypes so that
#          it remains 0 or 1, otherwise it will
#          be transformed to boolean feature.
#          (its ok too).
data = data.replace(mappings)

# Get worst state for patient
data = data.groupby(by="study_no", dropna=False) \
    .agg(aggregation).dropna()

# Show data
print("\nData:")
print(data)
print(data.dtypes)
print(data.index.nunique())
print(data[outcomes].sum().sort_values())


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
# .. note: The scaling method should have also been
#          saved when training the model.
# Features
datap = data[features].copy(deep=True)

# Scale first
datap = MinMaxScaler().fit_transform(datap)

# Encode
encoded = model.encode_inputs( \
    DataLoader(datap, 16, shuffle=False))

# Include in original dataset
data[['x', 'y']] = encoded

# Show
print("\nEncoded:")
print(encoded)
print("\nData:")
print(data)

# -------------------------------
# Visualization
# -------------------------------
# Libraries
import matplotlib.pyplot as plt

# Gaussian kde
from scipy.stats import gaussian_kde

# Own
from _utils import kde_mpl_compute
from _utils import kde_mpl_plot
from _utils import kde_mpl

# Data
x, y, z = data.x, data.y, data.plt


# --------------------------
# Direct plotting
# --------------------------
# Plot hexbin
fig, axes = plt.subplots(ncols=3, figsize=(15, 4), sharey=True)

# Plot count
m1 = axes.flat[0].hexbin(x, y, label=c,
    gridsize=30, cmap=cmaps.get(c, 'Reds'))
axes.flat[0].set(title='count', aspect='equal')
plt.colorbar(m1, ax=axes.flat[0])

# Plot mean
m2 = axes.flat[1].hexbin(x, y, C=z, label=c,
    gridsize=30, cmap=cmaps.get(c, 'Reds'))
axes.flat[1].set(title='mean', aspect='equal')
plt.colorbar(m2, ax=axes.flat[1])

# Create contours
kde_mpl(x, y, weights=z, ax=axes.flat[2])
axes.flat[2].set(title='contours (not working)', aspect='equal')

# Configure
plt.tight_layout()

# --------------------------
# Using hist2d functions
# --------------------------
# Create figure
fig, (ax1, ax2, ax3, ax4) = \
    plt.subplots(ncols=4, figsize=(15, 3), sharey=True)

# Compute count
counts, xbins, ybins, im1 = \
    ax1.hist2d(x, y, bins=(30, 30), cmap='Reds')

# Compute sums
sums, _, _, im2 = \
    ax2.hist2d(x, y, weights=z,
               bins=(xbins, ybins),
               cmap='Reds')

# Compute mean
mean = (sums / counts).T

# Plot colormesh
with np.errstate(divide='ignore', invalid='ignore'):
    m3 = ax3.pcolormesh(ybins, xbins,
        mean, cmap='Reds')

# .. note: We are ignoring the last bin value.
# Plot contours
ax4.contour(xbins[:-1], ybins[:-1], mean,
    levels=14, linewidths=0.25, alpha=0.5,
    linestyles='dashed', colors='k')
cntr = ax4.contourf(xbins[:-1], ybins[:-1], mean,
    levels=14, cmap='Reds')
cb = plt.colorbar(cntr, ax=ax4)

# Plot image
#ax5.imshow(mean, origin='lower',
#                 aspect='auto',
#                 cmap='Reds')

# Configure
ax1.set(aspect='equal', title='counts')
ax2.set(aspect='equal', title='sum')
ax3.set(aspect='equal', title='mean (colormesh)')
ax4.set(aspect='equal', title='mean (contour)')
#ax5.set(aspect='equal', title='mean (imshow)')

# ---------------------------
# Using histogram2d functions
# ---------------------------
# Compute counts
counts, xbins, ybins = \
    np.histogram2d(x, y, bins=(30, 30))

# Compute sums
sums, _, _ = \
    np.histogram2d(x, y, weights=z, bins=(xbins, ybins))

# Compute mean
mean = (sums / counts).T

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,
    figsize=(15, 4), sharey=True)

# Plot counts
m1 = ax1.pcolormesh(xbins, ybins, counts.T, cmap='Reds')
plt.colorbar(m1, ax=ax1)

# Plot sums
m2 = ax2.pcolormesh(xbins, ybins, sums.T, cmap='Reds')
plt.colorbar(m2, ax=ax2)

# Plot mean
with np.errstate(divide='ignore', invalid='ignore'):  
    m3 = ax3.pcolormesh(xbins, ybins,
        mean, cmap='Reds')
plt.colorbar(m3, ax=ax3)

# Configure
ax1.set(aspect='equal', title='counts')
ax2.set(aspect='equal', title='sum')
ax3.set(aspect='equal', title='mean')

# Configure
plt.tight_layout()

# Show
plt.show()