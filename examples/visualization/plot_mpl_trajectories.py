"""
Trajectories
============

This script is useful to understand how the latent space
can be used to visualise the progression (trajectories)
of patients over time.

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
    "mild"]

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
    data.bleeding_urine | \
    data.bleeding_mucosal

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

# Raw after very basic cleaning
raw = data.copy(deep=True)

# Filter data (age, iqr, ...)
# .. note: Although previously we were filtering only
#          for adults, there is an interesting patient
#          (01nva-003-2164) who is 19 years old.
#
# .. note: Platelets has a very clear outlier 50000
#          which influences the quartile selection
#          in the quartile range rule (IQR)
data = data[data.age.between(0.0, 18.0)]
data = data[data.plt < 50000] # extreme outlier

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
f1, axes1 = plt.subplots(1, 3,
    figsize=(15, 4), sharex=True, sharey=True)

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

# Plot aggregated (data_w)
aux = data_w[data_w.dsource == '01nva']
axes1.flat[0].scatter(aux.x, aux.y,
    s=4, linewidth=0.5,
    c=aux.shock \
        .replace({0:'k', 1:'r'}) \
        .fillna('k'))

# Plot daily (data_f)
aux = data_f[data_f.dsource == '01nva']
axes1.flat[1].scatter(aux.x, aux.y,
    s=4, linewidth=0.5,
    c=aux.event_shock \
        .replace({0:'k', 1:'r'}) \
        .fillna('k'))

# Order by study_no and day_from_admission
aux = aux.sort_values( \
    by=['study_no', 'day_from_admission'])

# Plot trajectories
for i,g in aux.groupby('study_no'):
    axes1.flat[2].plot(g.x, g.y, c='k',
       marker='o', markersize=2, linewidth=0.5)
    for i,j,v in zip(g.x, g.y, g.day_from_admission):
        axes1.flat[2].annotate(str(v), xy=(i, j))

# Configure
plt.tight_layout()



# -------------------------------------
# Individual trajectories
# -------------------------------------
# .. note: Note that the scaling method before considers
#          only patients under 18. Thus, wen testing patients
#          over 18 they are pushed to the right hand side.
#          Thus, we are min max scaling with all ages although
#          this is not ideal if algorithm was trained with <18.
# Scaler
scaler = MinMaxScaler().fit(data_f[features])
#scaler = MinMaxScaler().fit(raw[features])

# Convert dtypes
data = data.convert_dtypes()

# Select all 01nva data and clean
data = raw[raw.dsource.isin(['01nva', '06dx'])] \
    .copy(deep=True) \
    .dropna(how='any', subset=features) \
    .sort_values(by=['study_no',
                     'day_from_admission'])

# Include encoded in full (filtered)
data[['x', 'y']] = model.encode_inputs( \
    DataLoader(scaler.transform(data[features]),
         16, shuffle=False))

# Compute KDE (children)
aux = data_w[data_w.severe == 1]
xgrid, ygrid, Zgrid = kde_mpl_compute(aux.x, aux.y,
    xlim=(data_w.x.min(), data_w.x.max()),
    ylim=(data_w.y.min(), data_w.y.max()))

"""
# Compute KD2 (all)
# Get worst state for patient
scl = MinMaxScaler().fit(raw[features])
aux = raw.copy(deep=True)
aux_w = aux.copy(deep=True) \
    .groupby(by="study_no", dropna=False) \
    .agg(aggregation) \
    .dropna(how='any', subset=features)
# Include encoded in full (filtered)
aux_w[['x', 'y']] = model.encode_inputs( \
    DataLoader(scl.transform(aux_w[features]),
         16, shuffle=False))
aux = aux_w[aux_w.severe == 1]
xgrid, ygrid, Zgrid = kde_mpl_compute(aux.x, aux.y,
    xlim=(aux_w.x.min(), aux_w.x.max()),
    ylim=(aux_w.y.min(), aux_w.y.max()))
"""
# Show
print("\nPatients:")
print(data.study_no.nunique())

##############################################
# Plot specific patients
# ----------------------

# -------------------------
# Plot interesting patients
# -------------------------
# Lets define the patients

patients = [
    1105,
    2012, # 2 shocks
    2013,
    2026, # 3 shocks
    2103, # 2 shocks
    2104,
    2110,
    2168,
    2203,
    2205,
    2206,
    2205,
    2217,
    2222,
    2207, # 3 shocks
    2209, # 2 shocks
]

p1 = ['01nva-003-%s' % n for n in patients]
p2 = ['06dx-06dxa249']

patients = p1 + p2

print(sorted(data[data.dsource=='06dx'].study_no.unique()))

# Plot graph for each patient
for p in patients:
    # Get data
    #aux = data[data.study_no.str.endswith(str(p))]

    aux = data[data.study_no.str.lower().str.contains(p)]

    if aux.shape[0] == 0:
        continue

    # Create figure
    f, ax = plt.subplots(1, 1)

    # Plot KDE (data aggregated)
    kde_mpl_plot(xgrid, ygrid, Zgrid, ax=ax,
        contour=True, cbar=False, cmap='Reds')

    # Plot evolution (line + markers)
    ax.plot(aux.x, aux.y, c='k', label=p, alpha=0.75,
        marker='o', markersize=5, linewidth=0.5)
    ax.set(title='Patient %s (%s years)' % \
        (str(p), int(aux.age.values[0])), aspect='equal')

    # Plot numbers
    for i,j,v in zip(aux.x, aux.y, aux.day_from_admission):
        ax.annotate(str(int(v)), xy=(i+0.01, j))

    # Plot shock
    if aux.event_shock.any():
        ax.scatter(
            aux[aux.event_shock.fillna(False)].x,
            aux[aux.event_shock.fillna(False)].y,
            marker='o', color='red', edgecolor='k',
            s=10, linewidth=0.5, zorder=10)

    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    plt.tight_layout()

    # Show
    #print(aux[features + others + ['shock', 'event_shock']])


plt.show()

import sys
sys.exit()
####################################################
# Plot all trajectories
# ---------------------

# -------------------------
# Plot ALL patients
# -------------------------
# Define rows and columns
nrows, ncols = 3, 5

# Filter more than length n
v = data.study_no.value_counts()
aux = data[data.study_no.isin(v.index[v.gt(0)])]

# Create groups
groups = aux.groupby('study_no')

# Define rows and columns
nrows, ncols = 3, 5

# For each patient
for i, (n,g) in enumerate(groups):

    # Compute idx
    idx = i % (nrows*ncols)

    # Create figure
    if (idx == 0):
        # Adjust axes
        if i > 0:
            plt.tight_layout()
        # Create figure
        f, axes = plt.subplots(nrows, ncols,
            figsize=(ncols * 3.15, nrows * 2.5),
            sharex=True, sharey=True)

    # Get axes
    ax = axes.flat[idx]

    # Plot KDE (data aggregated)
    kde_mpl_plot(xgrid, ygrid, Zgrid, ax=ax,
        contour=True, cbar=True, cmap='Reds')

    # Plot line and markers
    ax.plot(g.x, g.y, c='k', label=n,
        marker='o', markersize=3, linewidth=0.5)
    ax.set(title=n.split('-')[-1], aspect='equal')
    ax.set(title='%s (%s yo)' % \
        (str(p), int(g.age.values[0])), aspect='equal')

    # Highlight shocks
    if g.event_shock.any():
        ax.scatter(
            g[g.event_shock.fillna(False)].x,
            g[g.event_shock.fillna(False)].y,
            marker='o', color='red', edgecolor='k',
            s=10, linewidth=0.5, zorder=10)

    # Plot numbers
    for k,j,v in zip(g.x, g.y, g.day_from_admission):
        ax.annotate(str(int(v)), xy=(k+0.05, j))
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Show legend
    #ax.legend(loc="lower right")

# Configure
plt.tight_layout()

# Show
plt.show()