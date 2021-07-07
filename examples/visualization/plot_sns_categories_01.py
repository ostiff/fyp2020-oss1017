"""
Seaborn (incomplete!)
=====================

Trying to display things with seaborn.

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
    "mild",
]

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
    'date': 'last',
    'age': 'max',
    'weight': 'mean',
    'plt': 'min',
    'haematocrit_percent': 'max',
    'body_temperature': 'mean',
    'gender': 'first',
}

cmaps = {
    'warning': 'Oranges',
    'severe': 'Reds',
    'mild': 'Blues'
}

nrows, ncols = 1, 3

# ------------------------------
# Load data
# ------------------------------
# Load data
data = pd.read_csv('resources/datasets/combined_tidy_v0.0.10.csv')

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

# Ensure all outcomes are in aggregation
for c in outcomes:
    if c not in aggregation:
        print("Adding... %23s | max" % c)
        aggregation[c] = 'max'

# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000]

# Filter outliers
data = IQR_rule(data, [
    'plt',
])

# Rename
# .. note: Done after convert_dtypes so that
#          it remains 0 or 1, otherwise it will
#          be transformed to boolean feature.
#          (its ok too).
data = data.replace(mappings)

# Show dtypes
print("\nDtypes:")
print(data.dtypes)

# Get worst state for patient
data = data.groupby(by="study_no", dropna=False) \
    .agg(aggregation).dropna()

# Show data
print("\nData:")
print(data)
print(data.index.nunique())
print(data[outcomes].sum().sort_values())


# ------------------------------
# Load Model
# ------------------------------
# Load model
model_path = 'resources/models/ae_sig_3'
model = pickle.load(open(model_path, 'rb'))

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

# Show
print("\nEncoded:")
print(encoded)

# Include in original dataset
data[['x', 'y']] = encoded

# Show
print("\nData:")
print(data)


# ------------------------------
# Visualization
# ------------------------------
# Libraries
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------
# Show interesting outcomes (contours)
# ------------------------------------
# Figure with kdes
f5, axes5 = plt.subplots(nrows, ncols, figsize=(15, 4),
    sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Select dataset
    aux = data[data[c] == 1]
    # Plot kde
    sns.kdeplot(x=aux.x, y=aux.y,
        ax=axes5.flat[i], levels=14,
        fill=True, palette=mpl.cm.get_cmap('Reds')
    )
    # Configure
    axes5.flat[i].set(aspect='equal',
        title='%s (%s)' % (c, aux.shape[0]))

# Configure
plt.tight_layout()

# Show
plt.show()