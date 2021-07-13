"""
Outcomes (boolean)
==================

This script is useful to understand the distribution
over the 2D projected space of the different outcomes.
For that purpose, first patients are grouped to their
worst case scenario (highest HCT and lowest PLT). Then
the number of patients with certain outcome in the 2D
space is calculate (histogram) and plotted.

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
    "severe",
    "warning",
    "mild",
    "shock",
    "severe_leak",
    "severe_bleed",
    "severe_organ",
    "overload",
    "ascites",
    "oedema",
    "effusion",
    "pleural_effusion",
    "jaundice",
    "liver_abnormal",
    "cns_abnormal",
    #"cerebro", #0
    "neurology",
    #"kidney_abnormal",#0
    "bleeding_gi",
    #"bleeding_urine", #0
    "bleeding_mucosal",
    "bleeding_skin",
    "bleeding_gum",
    "bleeding_nose",
    "respiratory_distress",
    "vomiting",
    "abdominal_pain",
    "abdominal_tenderness",
    "chest_indrawing",
    "event_death",
    "heart_sound_abnormal",
    #"inotrope", 0
    "lung",
    #"perfusion", #0
    "pericardial_effusion",
    "previous_dengue",
    #"renal_disease", #0
    #"xray_pleural_effusion", #0
    "restlessness",
    "lethargy",
    "alt>1000",
    "ast>1000",
    "liver>2",
    "pcr_dengue_load"
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

#
print(data.convert_dtypes().select_dtypes(include=['bool']).sum(axis=0).sort_values())

data.convert_dtypes().select_dtypes(include=['bool']).sum(axis=0).sort_values().to_csv('aux.csv')

# Conditions
data['ast>1000'] = data.ast.fillna(0) >= 1000
data['alt>1000'] = data.alt.fillna(0) >= 1000
data['liver>2'] = data.liver_palpation_size.fillna(0) > 2

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

# .. note: We are assuming that these are all
#          boolean features, usually containing
#          1 when an unusual event happens. Thus
#          filling with 0 (e.g. shock).
#
# .. note: What if the information was not
#          collected in the whole study? We are
#          only plotting kde for True (1) value
#          so no problem for zeros.
# Fill empty values
for c in outcomes:
    data[c] = data[c].fillna(0)

# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000] # extreme outlier
#data = data[data.ast < 1500]  # extreme outlier

# Filter outliers
data = IQR_rule(data, [
    'plt',
])

# Needs all features for projection
data = data.dropna(how='any', subset=features)

# Rename
# .. note: Done after convert_dtypes so that
#          it remains 0 or 1, otherwise it will
#          be transformed to boolean feature.
#          (its ok too).
data = data.replace(mappings)

# .. note: aggregating by patient so we have
#          one single outcome per patient.
# Get worst state for patient
data = data.groupby(by="study_no", dropna=False) \
    .agg(aggregation).dropna()

# Show data
print("\nData:")
print(data)
print(data.dtypes)
print(data.index.nunique())
print(data[outcomes].sum() \
    .sort_values(ascending=False))

print("\nCount:")
print(data.convert_dtypes() \
    .select_dtypes('bool').sum() \
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

####################################################
# Display configuration
# ---------------------

####################################################
# Display using contours
# ----------------------
# Define rows and columns
nrows, ncols = 3, 5

# Titles
titles = {
    'cns_abnormal': 'CNS Abnormal',
    'pcr_dengue_load': 'PCR Dengue Load'
}


# ------------------------------------
# Show interesting outcomes (contours)
# ------------------------------------
# For each outcome
for i, o in enumerate(outcomes):

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

    # Select dataset
    aux = data[data[o] == 1]
    # Plot kde
    kde_mpl(aux.x, aux.y, ax=axes.flat[idx],
        contour=True, cmap=cmaps.get(o, 'Reds'),
        xlim=(data.x.min(), data.x.max()),
        ylim=(data.y.min(), data.y.max()))
    # Configure
    axes.flat[idx].set(aspect='equal',
        xlim=(data.x.min(), data.x.max()),
        ylim=(data.y.min(), data.y.max()),
        title='%s (%s)' % (titles.get(o, o) \
            .replace('_', ' ').title(),
            aux[o].shape[0]))

# Configure
plt.tight_layout()


####################################################
# Display using hexbins
# ----------------------

# -----------------------------------
# Show interesting outcomes (hexbins)
# -----------------------------------
# For each outcome
for i, o in enumerate(outcomes):

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

    # Select dataset
    aux = data[data[o] == 1]
    # Plot hexbin
    m = axes.flat[idx].hexbin(aux.x, aux.y,
        label=o, gridsize=30, marginals=False,
        cmap=cmaps.get(o, 'Reds'))
    # Configure
    axes.flat[idx].set(aspect='equal',
        xlim=(data.x.min(), data.x.max()),
        ylim=(data.y.min(), data.y.max()),
        title='%s (%s)' % (titles.get(o, o) \
            .replace('_', ' ').title(),
            aux[o].shape[0]))
    # Colorbar
    plt.colorbar(m, ax=axes.flat[idx])

# Configure
plt.tight_layout()



####################################################
# Display using hexbins
# ----------------------

# -----------------------------------
# Show interesting outcomes (hexbins)
# -----------------------------------
# For each outcome
for i, o in enumerate(outcomes):

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

    # Select dataset
    aux = data[data[o] == 1]
    # Plot kde
    kde_mpl(aux.x, aux.y, ax=axes.flat[idx],
            contour=False, cmap=cmaps.get(o, 'Reds'),
            xlim=(data.x.min(), data.x.max()),
            ylim=(data.y.min(), data.y.max()))
    # Configure
    axes.flat[idx].set(aspect='equal',
        xlim=(data.x.min(), data.x.max()),
        ylim=(data.y.min(), data.y.max()),
        title='%s (%s)' % (titles.get(o, o) \
            .replace('_', ' ').title(),
            aux[o].sum()))
    # Colorbar
    plt.colorbar(m, ax=axes.flat[idx])

# Configure
plt.tight_layout()

# Show
plt.show()