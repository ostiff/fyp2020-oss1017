"""
Kde
===
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

# Raw after very basic cleaning
raw = data.copy(deep=True)

# Filter data (age, iqr, ...)
# .. note: Although previously we were filtering only
#          for adults, there is an interesting patient
#          (01nva-003-2164) who is 19 years old.
#
# .. note: Platelets has a very clear outliers 50000
#          which influences the quartile selection
#          in the quartile range rule (IQR)

data = data[data.age.between(0.0, 20.0)]
data = data[data.plt < 50000]
#data = data.fillna('ffill') # no need if keeping max/min
#data = data.fillna('bfill') # no need if keeping max/min

# Filter outliers
data = IQR_rule(data, [
    'plt',
    #'haematocrit_percent',
    #'body_temperature'
])

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
from _utils import kde_mpl_plot
from _utils import kde_mpl_compute
from _utils import kde_mpl

# ------------------------------------
# Show interesting outcomes (contours)
# ------------------------------------
# Figure with kdes
f1, axes1 = plt.subplots(1, 3, figsize=(15, 4),
    sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Select dataset
    aux = data_w[data_w[c] == 1]
    # Plot kde
    kde_mpl(aux.x, aux.y, ax=axes1.flat[i],
        contour=True, cmap=cmaps.get(c, 'Reds'),
        xlim=(data_w.x.min(), data_w.x.max()),
        ylim=(data_w.y.min(), data_w.y.max()))
    # Configure
    axes1.flat[i].set(aspect='equal',
        title='%s (%s)' % (c.title(), aux.shape[0]))



# -------------------------------------
# Plot aggregated categories
# -------------------------------------
# Copy 01nva data
aux = data_w[data_w.dsource == '01nva']
aux['study_no'] = aux.index \
    .to_series().str.split('-').str[-1]

# Create basic rule
criteria = list(zip(*[
    (aux.study_no.between('2001', '2034'), 'Severe'),
    (aux.study_no.between('2101', '2112'), 'Severe'),
    (aux.study_no.between('2201', '2236'), 'Severe'),
    (aux.study_no.between('2151', '2196'), 'Mild'),
    (aux.study_no == '2162', 'Unique'),
]))

# Apply rule (default False)
aux.loc[:, 'category'] = np.select(criteria[0],
                                   criteria[1],
                                   'None')

# Display
for i,g in aux.groupby('category'):
    axes1.flat[0].scatter(g.x, g.y,
        s=8, linewidth=0.5, edgecolor='k',
        label=i)
    axes1.flat[0].legend()


# -------------------------------------
# Plot daily categories
# -------------------------------------
# Copy 01nva data
aux = data_f[data_f.dsource == '01nva']
aux['study_no'] = aux.study_no \
    .str.split('-').str[-1]

# Create basic rule
criteria = list(zip(*[
    (aux.study_no.between('2001', '2034'), 'Severe'),
    (aux.study_no.between('2101', '2112'), 'Severe'),
    (aux.study_no.between('2201', '2236'), 'Severe'),
    (aux.study_no.between('2151', '2196'), 'Mild'),
    (aux.study_no == '2162', 'Unique'),
]))

# Apply rule (default False)
aux.loc[:, 'category'] = np.select(criteria[0],
                                   criteria[1],
                                   'None')

# Display
for i,g in aux.groupby('category'):
    axes1.flat[1].scatter(g.x, g.y,
        s=8, linewidth=0.5, edgecolor='k',
        label=i)
    axes1.flat[1].legend()

# -------------------------------------
# Plot all patients
# -------------------------------------
# Select all 01nva data and clean
aux = raw[raw.dsource == '01nva'] \
    .copy(deep=True) \
    .dropna(how='any', subset=features) \
    .sort_values(by=['study_no',
                     'day_from_admission'])

# Include encoded in full (filtered)
aux[['x', 'y']] = model.encode_inputs( \
    DataLoader(scaler.transform(aux[features]),
         16, shuffle=False))
# Add study_no
aux['study_no'] = aux.study_no \
    .str.split('-').str[-1]

# Issue with >18 because model trained only with children.
aux = aux[aux.age <= 18]

# Create basic rule
criteria = list(zip(*[
    (aux.study_no.between('2001', '2034'), 'Severe'),
    (aux.study_no.between('2101', '2112'), 'Severe'),
    (aux.study_no.between('2201', '2236'), 'Severe'),
    (aux.study_no.between('2151', '2196'), 'Mild'),
    (aux.study_no == '2162', 'Unique'),
]))

# Apply rule (default False)
aux['category'] = np.select(criteria[0],
                            criteria[1],
                            'None')

# Display
for i,g in aux.groupby('category'):
    axes1.flat[2].scatter(g.x, g.y,
        s=8, linewidth=0.5, edgecolor='k',
        label=i)
    axes1.flat[2].legend()

# Configure
plt.tight_layout()

# Show
plt.show()