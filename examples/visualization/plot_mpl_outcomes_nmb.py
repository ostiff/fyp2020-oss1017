"""
Outcomes (numbers)
==================

This script is useful to understand the distribution
over the 2D projected space of the different numeric
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
    "age",
    "weight",
    "plt",
    "haematocrit_percent",
    "body_temperature",
    "ast",
    "alt"
]

# Ensure all outcomes are in aggregation
for c in outcomes:
    if c not in aggregation:
        print("Adding... %23s | max" % c)
        aggregation[c] = 'max'

# Define rows and columns
nrows, ncols = 3, 3

# ------------------------------
# Load data
# ------------------------------
# Load data
data = _utils.load_data()

# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000] # extreme outlier
data = data[data.ast < 1500]  # extreme outlier

# Filter outliers
data = IQR_rule(data, [
    'plt',
    #'haematocrit_percent',
    #'body_temperature'
])

# Show data
print("\nData:")
print(data)
print(data.dtypes)
print(data.index.nunique())
print(data[outcomes].count().sort_values())


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
# .. note: Ideally the method to preprocess
#          data should be included in the
#          model (like a sklearn pipeline)
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


# ------------------------------
# Visualization
# ------------------------------
# Libraries
import matplotlib.pyplot as plt

# Figure with hexbins
f3, axes3 = plt.subplots(nrows, ncols,
    figsize=(12, 9), sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Plot hexbin
    m = axes3.flat[i].hexbin(data.x, data.y,
        C=data[c], label=c, gridsize=30,
        cmap=cmaps.get(c, 'Reds'))
    # Configure
    axes3.flat[i].set(aspect='equal',
        title='%s (%s)' % (c, data[c].count()))
    plt.colorbar(m, ax=axes3.flat[i])

# Configure
plt.tight_layout()

# Show
plt.show()