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
    "shock",
    #"ascites",
    #"overload",
    #"oedema_pulmonary",
    #"respiratory_distress",
    #"ventilation",
    #"diuretics",
    #"bleeding_gi",
    #"bleeding_urine",
    #"bleeding_severe",
    #"bleeding_mucosal",
    #"bleeding_skin",
    #"vomiting", # is this vomiting blood?
    #"abdominal_pain",
    #"cns_abnormal",
    "sd_leak",
    "sd_bleed",
    "sd_organ",
    "gender",
    "severe"]

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
    'shock': 'max',
    'sd_leak': 'max',
    'sd_bleed': 'max',
    'sd_organ': 'max',
    'date': 'last',
    'mild': 'max',
    'severe': 'max',
    'warning': 'max'
    #'bleeding_gi': 'max',
    #'bleeding_urine': 'max',
    #'vomiting': 'max',
    #'abdominal_pain': 'max',
    #'bleeding_mucosal': 'max',
    #'bleeding_skin': 'max'
}


# ------------------------------
# Load data
# ------------------------------
# Load data
data = pd.read_csv('resources/datasets/combined_tidy_v0.0.10.csv')

# Select columns
#data = data[features + outcomes + others]

# Create features
data['sd_leak'] = \
    data.ascites | \
    data.overload | \
    data.oedema_pulmonary | \
    data.respiratory_distress

data['sd_bleed'] = \
    data.bleeding_gi | \
    data.bleeding_urine

data['sd_organ'] = \
    data.cns_abnormal

data['warning']= \
    data.abdominal_pain | \
    data.vomiting | \
    data.ascites | \
    data.pleural_effusion | \
    data.bleeding_mucosal | \
    data.restlessness | \
    data.lethargy
    #data.agitated | \

data['severe'] = \
    data.sd_bleeding | \
    data.sd_leakage | \
    data.sd_organ_impairment | \
    data.shock

data['mild'] = \
    ~(data.severe | data.warning)


# Filter data (age, iqr, ...)
data = data[data.age.between(0.0, 18.0)]
#data = data.fillna('ffill') # no need if keeping max/min
#data = data.fillna('bfill') # no need if keeping max/min

# Filter outliers
data = IQR_rule(data, [
    'plt',
    'haematocrit_percent',
    'body_temperature'
])

# Fill empty values
data = data.fillna({
    'sd_leak': 0,
    'sd_bleed': 0,
    'sd_organ': 0
})

# Convert dtypes
data = data.convert_dtypes()

# Rename
# .. note: Done after convert_dtypes so that
#          it remains 0 or 1, otherwise it will
#          be transformed to boolean feature.
#          (its ok too).
data = data.replace(mappings)

data_full = data.copy(deep=True)

# Drop nan
#data = data.dropna(how='any',
#   subset=features + outcomes)

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
print(data[outcomes].sum())

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
import matplotlib.pyplot as plt

# Specific
from scipy.stats import gaussian_kde


def kde_mpl(x, y, cmap='Reds', ax=None, contour=True):
    """Plot gaussian kde matplotlib

    Parameters
    ----------
    x, y: arrays
        Numpy arrays with the 2D values.

    Returns
    -------

    """
    try:
        # Plot density
        kde = gaussian_kde(np.vstack((x, y)))

        # Parameters
        xmin, xmax = min(data.x), max(data.x)
        ymin, ymax = min(data.y), max(data.y)

        # evaluate on a regular grid
        xgrid = np.linspace(xmin, xmax, 100)
        ygrid = np.linspace(ymin, ymax, 100)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        zgrid = kde.evaluate(np.vstack([
            Xgrid.ravel(),
            Ygrid.ravel()
        ]))
        Zgrid = zgrid.reshape(Xgrid.shape)

        # Plot the result as an image
        ax.imshow(Zgrid,
            origin='lower', aspect='auto',
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap)

        # Plot contour
        if contour:
            ax.contour(xgrid, ygrid, Zgrid,
                levels=14, linewidths=0.25, alpha=0.5,
                linestyles='dashed', colors='k')
            cntr = ax.contourf(xgrid, ygrid, Zgrid,
                levels=14, cmap=cmap)
            cb = plt.colorbar(cntr, ax=ax)
            #cb.set_label('density')

        return ax
    except Exception as e:
        print("Exception! %s" % e)
        return ax


# Plot individual
INDIVIDUAL = False


# ---------------------
# Show all points in 2D
# ---------------------
# Create figure
f1, axes1 = plt.subplots(1, 2, figsize=(10, 5),
    sharex=True, sharey=True)

# Show 2D histogram and hexbin
f11 = axes1.flat[0].hist2d(data.x, data.y,
    bins=30, cmap='Blues')
f12 = axes1.flat[1].hexbin(data.x, data.y,
    gridsize=30, cmap='Blues')

# Titles
axes1.flat[0].set(title='All patients (%s) - sqr' % data.shape[0])
axes1.flat[1].set(title='All patients (%s) - hex' % data.shape[0])

# Configuration
for ax in axes1.flat:
    ax.set(aspect='equal')
    #cb = plt.colorbar(f, cax=ax)
    #ax.yaxis.tick_right()
    #ax.yaxis.set_tick_params(labelright=False)
    #cb.set_label('counts in bin (sqr)')


# --------------------------------
# Show interesting outcomes (hist)
# --------------------------------
# Figure with hexbins
f3, axes3 = plt.subplots(2, 4, figsize=(12, 5),
    sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Select dataset
    aux = data[data[c] == 1]
    # Plot hexbin
    axes3.flat[i].hexbin(aux.x, aux.y,
        gridsize=30, cmap='Reds',
        label=c)
    # Configure
    axes3.flat[i].set(aspect='equal',
        title='%s (%s)' % (c, aux.shape[0]))

    if INDIVIDUAL:
        plt.figure()
        plt.hexbin(aux.x, aux.y, gridsize=30, cmap='Reds')
        cb = plt.colorbar()
        cb.set_label('counts in bin (hex)')
        plt.title('%s' % c)

# Configure
plt.suptitle('Overview - hexbin')
plt.tight_layout(w_pad=1, h_pad=1)

# --------------------------------
# Show interesting outcomes (KDE)
# --------------------------------
# Figure with kdes
f4, axes4 = plt.subplots(2, 4, figsize=(12, 5),
    sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Select dataset
    aux = data[data[c] == 1]
    # Plot kde
    kde_mpl(aux.x, aux.y, ax=axes4.flat[i], contour=False)
    # Configure
    axes4.flat[i].set(aspect='equal',
        title='%s (%s)' % (c, aux.shape[0]))

# Plot rest
aux = data[data.mild]
kde_mpl(aux.x, aux.y, ax=axes4.flat[-1],
    cmap='Blues', contour=False)
axes4.flat[-1].set(aspect='equal',
    title='%s (%s)' % ('Mild', aux.shape[0]))

# Plot warning
aux = data[data.warning]
kde_mpl(aux.x, aux.y, ax=axes4.flat[-2],
    cmap='Oranges', contour=False)
axes4.flat[-2].set(aspect='equal',
    title='%s (%s)' % ('Warning', aux.shape[0]))

# Configure
plt.suptitle('Overview - kdes')
plt.tight_layout(w_pad=1, h_pad=1)



# ------------------------------------
# Show interesting outcomes (contours)
# ------------------------------------
# Figure with kdes
f5, axes5 = plt.subplots(2, 4, figsize=(12, 5),
    sharex=True, sharey=True)

# Loop
for i, c in enumerate(outcomes):
    # Select dataset
    aux = data[data[c] == 1]
    # Plot kde
    kde_mpl(aux.x, aux.y, ax=axes5.flat[i],
        contour=True)
    # Configure
    axes5.flat[i].set(aspect='equal',
        title='%s (%s)' % (c, aux.shape[0]))

# Plot rest
aux = data[data.mild]
kde_mpl(aux.x, aux.y, ax=axes5.flat[-1],
    cmap='Blues', contour=True)
axes5.flat[-1].set(aspect='equal',
    title='%s (%s)' % ('Mild', aux.shape[0]))

# Plot warning
aux = data[data.warning]
kde_mpl(aux.x, aux.y, ax=axes5.flat[-2],
    cmap='Oranges', contour=True)
axes5.flat[-2].set(aspect='equal',
    title='%s (%s)' % ('Warning', aux.shape[0]))


# Configure
plt.suptitle('Overview - kdes')
plt.tight_layout(w_pad=1, h_pad=1)


def to_studyno(x):
    return set(['01nva-003-%s' % i for i in x])

developed = to_studyno([2162])
nonsevere = to_studyno(list(range(2151, 2196)))
severe = to_studyno(\
    list(range(2101, 2112)) + \
    list(range(2201, 2236)) + \
    list(range(2001, 2034)))


groups = data[data.dsource == '01nva'].copy(deep=True)
#groups = groups.reset_index()
groups['class'] = None
groups.loc[developed.intersection(groups.index), 'class'] = 'developed'
groups.loc[nonsevere.intersection(groups.index), 'class'] = 'nonsevere'
groups.loc[severe.intersection(groups.index), 'class'] = 'severe'

# ------------------------------
# Projections
# ------------------------------
# .. note: The scaling method should have also been
#          saved when training the model.
# Features
dataf = data_full \
    .copy(deep=True) \
    .dropna(how='any', subset=features)
datap = dataf[features].copy(deep=True)



print(datap)

# Scale first
datap = MinMaxScaler().fit_transform(datap)

# Encode
encoded = model.encode_inputs( \
    DataLoader(datap, 16, shuffle=False))

# Include in original dataset
dataf[['x', 'y']] = encoded

data_aux = dataf[dataf.dsource == '01nva'].copy(deep=True)




print(data_aux)

# ------------------------------------
# Show interesting outcomes (contours)
# ------------------------------------
# Figure with kdes
f6, axes6= plt.subplots(1, 3, figsize=(12, 5),
    sharex=True, sharey=True)

# Plot rest
aux = data[data.severe]
kde_mpl(aux.x, aux.y, ax=axes6.flat[0],
    cmap='Reds', contour=True)
axes6.flat[0].set(aspect='equal',
    title='%s (%s)' % ('Severe', aux.shape[0]))

sev = aux.copy(deep=True)

# Plot rest
aux = data[data.mild]
kde_mpl(aux.x, aux.y, ax=axes6.flat[1],
    cmap='Blues', contour=True)
axes6.flat[1].set(aspect='equal',
    title='%s (%s)' % ('Mild', aux.shape[0]))

mild = aux.copy(deep=True)


axes6.flat[0].scatter(groups.x, groups.y, s=2)

axes6.flat[1].scatter(data_aux.x, data_aux.y, s=2, c='k')

aux = data_aux[data_aux.study_no.isin(severe)]

for i,g in aux.groupby('study_no'):
    print(g)
    g['date'] = pd.to_datetime(g.date)
    g['day'] = (g.date - g.date.min()).dt.days
    print(g.day)
    axes6.flat[2].plot(g.x, g.y, c='k',
       marker='o', markersize=2, linewidth=0.5)
    for i,j,v in zip(g.x, g.y, g.day):
        axes6.flat[2].annotate(str(v), xy=(i, j))


# -------------------------------------
# Individual trajectories
# -------------------------------------
# Create figure
f7, axes7 = plt.subplots(4, 8, figsize=(12, 5),
    sharex=True, sharey=True)

# For each patient
for c, (n,g) in enumerate(aux.groupby('study_no')):
    if c > 31:
        continue
    # Compute days
    g['date'] = pd.to_datetime(g.date)
    g['day'] = (g.date - g.date.min()).dt.days
    # Plot contours
    kde_mpl(sev.x, sev.y, ax=axes7.flat[c],
        cmap='Reds', contour=True)
    # Plot markers
    axes7.flat[c].plot(g.x, g.y, c='k',
       marker='o', markersize=2, linewidth=0.5)
    axes7.flat[c].set(title=n, aspect='equal')
    # Plot numbers
    for i,j,v in zip(g.x, g.y, g.day):
        axes7.flat[c].annotate(str(v), xy=(i+0.05, j))


plt.tight_layout()

#p = data_aux[data_aux.study_no.isin(['01nva-003-2028'])]
#print(p)
#axes6.flat[2].scatter(p.x, p.y, s=2, c='k')
#axes6.flat[2].plot(p.x, p.y, c='k',
#    marker='o', markersize=2, linewidth=0.5)

#for i, v in enumerate([1,2,3,4]):
#    axes6.flat[2].text(i, v+0.25, "%d" %v, ha="center")
#for v,(i,j) in enumerate(zip(p.x, p.y)):
#    axes6.flat[2].annotate(str(v), xy=(i,j))



plt.title("Severe - 01NVA agg")
plt.show()


"""
from sklearn.neighbors import KernelDensity
X = data[['x', 'y']]
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
scores = kde.score_samples(X)

print(scores)
"""
# Show
#plt.show()
