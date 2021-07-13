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
    "alt",
    "albumin",
    "creatine_kinase",
    "pcr_dengue_load",
    #"crp",     # too few
    #"glucose", # too few
    "haemoglobin",
    "height",
    "wbc",
    "neutrophils_percent",
    "lymphocytes_percent",
    "monocytes_percent",
    "neutrophils",
    "lymphocytes",
    "monocytes",
    "igm",
    "igg",
    "creatinine",
    "sbp",
    "dbp",
    "pulse",
    "liver_palpation_size",
    "spleen_palpation_size",
    "pulse_pressure",
    "pcr_dengue_reaction",
    "fibrinogen",
    "urea",
    "aptt",
    'hs_concentration',
    "potassium",
    "sodium"
]

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

# Replace
data.pcr_dengue_load.replace(0,
    value=np.nan, inplace=True)

# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000] # extreme outlier
data = data[data.ast < 1500]  # extreme outlier
#data = data[data.pcr_dengue_load < 1e10]

# Filter outliers
data = IQR_rule(data, [
    'plt',
])

# Needs all features for projection
data = data.dropna(how='any', subset=features)

# Show data
print("\nData:")
print(data)
print(data.dtypes)
print(data.index.nunique())
print(data[outcomes].describe())

# Show counts
print("\nCount:")
print(data.select_dtypes('number').count() \
    .sort_values(ascending=False).to_string())

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

# Latexify
mpl.rc('font', size=10)
mpl.rc('legend', fontsize=6)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)

# Titles
titles = {
    'haematocrit_percent': 'Haematocrit (%)',
    'monocytes_percent': 'Monocytes (%)',
    'lymphocytes_percent': 'Lymphocytes (%)',
    'neutrophils_percent': 'Neutrophils (%)',
    'plt': 'Platelets',
    'body_temperature': 'Temperature',
}

# Bins
bins = {
    'pcr_dengue_load': 'log',
}

# Define rows and columns
nrows, ncols = 3, 5

# Labels to drop
drop = [
    'day_from_onset',
    'day_from_enrolment',
    'day_from_admission',
    'x', 'y',
    'Unnamed: 0'
]

# Select outcomes
outcomes = data.convert_dtypes() \
    .select_dtypes(include=['float64', 'int64']) \
    .count().sort_values(ascending=False) \
    .drop(drop)
outcomes = outcomes[outcomes > 100]

# For each outcome
for i, o in enumerate(outcomes.index.tolist()):

    # Compute idx
    idx = i % (nrows*ncols)

    # Create figure
    if (idx == 0):
        # Adjust axes
        if i>0:
            plt.tight_layout()
        # Create figure
        f, axes = plt.subplots(nrows, ncols,
             figsize=(ncols * 3.15, nrows * 2.5),
             sharex=True, sharey=True)

    # Plot hexbin
    m = axes.flat[idx].hexbin(data.x, data.y,
        C=data[o], label=o, gridsize=30,
        bins=bins.get(o, None),
        cmap=cmaps.get(c, 'Reds'))
    # Configure
    axes.flat[idx].set(aspect='equal',
        xlim=(data.x.min(), data.x.max()),
        ylim=(data.y.min(), data.y.max()),
        title='%s (%s)' % (titles.get(o, o) \
            .replace('_', ' ').title(),
            data[o].count()))
    # Colorbar
    plt.colorbar(m, ax=axes.flat[idx])

# Configure
plt.tight_layout()

# Show
plt.show()


"""
# Loop
for i, c in enumerate(outcomes):
    # Plot hexbin
    m = axes3.flat[i].hexbin(data.x, data.y,
        C=data[c], label=c, gridsize=30,
        cmap=cmaps.get(c, 'Reds'))
    # Configure
    axes3.flat[i].set(aspect='equal',
        title='%s (%s)' % (titles.get(c, c) \
            .replace('_', ' ').title(),
            data[c].count()))
    plt.colorbar(m, ax=axes3.flat[i])
"""