"""
Dataset summary: Dengue -  grouped by patient
=============================================

Report generated using ``dataprep``.
"""

import pandas as pd
import numpy as np
from dataprep.eda import create_report
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.print_utils import suppress_stdout, suppress_stderr

features = ["dsource", "age", "gender", "weight", "bleeding", "plt",
            "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
            "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

with suppress_stdout() and suppress_stderr():

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

    df = IQR_rule(df, ['plt', 'haematocrit_percent', 'body_temperature'])

    report = create_report(df, title="Dengue dataset report")

report
