"""
Dengue dataset feature correlation
==================================

.. todo:: Add (boolean, boolean) correlation
.. todo:: Add (categorical, boolean/number) correlation
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from pkgname.utils.data_loader import load_dengue

features = ["dsource", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

df = load_dengue(usecols=['study_no']+features)

for feat in features:
    df[feat] = df.groupby('study_no')[feat].ffill().bfill()

df = df.loc[df['age'] <= 18]
df = df.dropna()

df = df.groupby(by="study_no", dropna=False).agg(
    dsource=pd.NamedAgg(column="dsource", aggfunc="last"),
    age=pd.NamedAgg(column="age", aggfunc="max"),
    gender=pd.NamedAgg(column="gender", aggfunc="first"),
    weight=pd.NamedAgg(column="weight", aggfunc=np.mean),
    bleeding=pd.NamedAgg(column="bleeding", aggfunc="max"),
    plt=pd.NamedAgg(column="plt", aggfunc="min"),
    shock=pd.NamedAgg(column="shock", aggfunc="max"),
    haematocrit_percent=pd.NamedAgg(column="haematocrit_percent", aggfunc="max"),
    bleeding_gum=pd.NamedAgg(column="bleeding_gum", aggfunc="max"),
    abdominal_pain=pd.NamedAgg(column="abdominal_pain", aggfunc="max"),
    ascites=pd.NamedAgg(column="ascites", aggfunc="max"),
    bleeding_mucosal=pd.NamedAgg(column="bleeding_mucosal", aggfunc="max"),
    bleeding_skin=pd.NamedAgg(column="bleeding_skin", aggfunc="max"),
    body_temperature=pd.NamedAgg(column="body_temperature", aggfunc=np.mean),
).dropna()

mapping = {'Female': True, 'Male': False}
rename_mapping = {'gender': 'female',
                  'haematocrit_percent': 'hct',
                  'body_temperature': 'b_temp'}
df = df.replace({'gender': mapping}).rename(columns=rename_mapping)

numerical = list(df.select_dtypes(include='number').columns)
boolean = list(df.select_dtypes(include='bool').columns)
categorical = list(df.select_dtypes(exclude=['bool', 'number']).columns)
columns = sorted(numerical) + sorted(boolean) + sorted(categorical)

# %%
# Continuous features
# -------------------
#
# Pearson correlation between continuous variables.

corr = df[numerical].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, square=True)

plt.show()


# %%
# Boolean features
# ----------------
#
# Point biserial correlation coefficient between boolean and
# continuous features

corr_data = [[pointbiserialr(df[feat_a], df[feat_b])[0]
              for feat_b in numerical] for feat_a in boolean]

corr = pd.DataFrame(data=corr_data, columns=numerical, index=boolean)
sns.heatmap(corr)

plt.show()


# %%
#
# Correlation coefficients between boolean features


# %%
# Categorical features
# --------------------
#
# X coefficient between categorical and other features

