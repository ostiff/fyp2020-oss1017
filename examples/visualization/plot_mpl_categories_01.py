"""
Categories (exhaustive)
=======================

This script is useful to understand the distribution
over the 2D projected space of the different categories.
The categories (severe, warning, mild, ...) can be
created by combining other existing features/outcomes
in the data. This script contains exhaustive, including
as much information as possible for each of the categories.


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

outcomes = [
    "severe",
    "warning",
    "mild",
]

# Ensure all outcomes are in aggregation
for c in outcomes:
    if c not in aggregation:
        print("Adding... %23s | max" % c)
        aggregation[c] = 'max'

nrows, ncols = 1, 3


# ------------------------------
# Load data
# ------------------------------
# Load data
data = _utils.load_data()

# Liver abnormal
data['liver_abnormal'] = \
    data.liver_acute | \
    data.liver_involved | \
    data.liver_failure | \
    data.liver_severe | \
    data.jaundice

# Kidney abnormal
data['kidney_abnormal'] = \
    data.skidney

# Create features
data['severe_leak'] = \
    data.ascites | \
    data.overload | \
    data.oedema_pulmonary | \
    data.respiratory_distress | \
    data.oedema | \
    data.pleural_effusion | \
    data.effusion

# Bleeding
data['severe_bleed'] = \
    data.bleeding_gi | \
    data.bleeding_urine # useless

# Organ impairment
data['severe_organ'] = \
    data.cns_abnormal | \
    data.neurology.astype(bool) | \
    data.liver_abnormal | \
    data.kidney_abnormal | \
    (data.ast.fillna(0) >= 1000) | \
    (data.alt.fillna(0) >= 1000)

# Category: severe
data['severe'] = \
    data.severe_leak | \
    data.severe_bleed | \
    data.severe_organ | \
    data.shock

# Category: warning WHO
data['warning'] = \
    data.abdominal_pain | \
    data.abdominal_tenderness | \
    data.vomiting | \
    data.ascites | \
    data.pleural_effusion | \
    data.bleeding_mucosal | \
    data.restlessness | \
    data.lethargy | \
    (data.liver_palpation_size.fillna(0) > 2)

# Category: mild
data['mild'] = ~(data.severe | data.warning)

# Fill empty values (be careful!)
for c in outcomes:
    data[c] = data[c].fillna(0)

# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000] # extreme outlier

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


# ------------------------------
# Visualization
# ------------------------------
# Libraries
import matplotlib.pyplot as plt

# Specific
from _utils import kde_mpl_plot
from _utils import kde_mpl_compute
from _utils import kde_mpl

# Figure with kdes
f5, axes5 = plt.subplots(nrows, ncols,
    figsize=(15, 4), sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Select dataset
    aux = data[data[c] == 1]
    # Plot kde
    kde_mpl(aux.x, aux.y, ax=axes5.flat[i],
        contour=True, cmap=cmaps.get(c, 'Reds'),
        xlim=(data.x.min(), data.x.max()),
        ylim=(data.y.min(), data.y.max()))
    # Configure
    axes5.flat[i].set(aspect='equal',
        title='%s (%s)' % (c, aux.shape[0]))

# Configure
plt.tight_layout()

# Show
plt.show()