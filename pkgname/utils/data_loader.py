#TODO: include scaling?

import pandas as pd
import numpy as np
from dataprep.eda import create_report

from definitions import DENGUE_PATH, PATHOLOGY_PATH


def remove_correlated(data, threshold, ignore):
    # Removed columns
    removed = set(ignore)

    # Pearson correlation matrix
    corr_matrix = data.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in removed):
                # Get name and add to removed set
                colname = corr_matrix.columns[i]
                removed.add(colname)

                # Remove column from dataset
                if colname in data.columns:
                    del data[colname]

    return data


def load_dengue(path=DENGUE_PATH, dropna=False, usecols=None, usedefault=False, removecorr=None):
    general_cols = ["study_no", "date", "age", "gender", "weight"]

    if usedefault:
        features = ["bleeding", "plt", "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                    "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]
        usecols = general_cols + features
        df = pd.read_csv(path, usecols=usecols, low_memory=False, parse_dates=['date'])

    else:
        if usecols and "date" in usecols:
            df = pd.read_csv(path, usecols=usecols, low_memory=False, parse_dates=['date'])
        else:
            df = pd.read_csv(path, usecols=usecols, low_memory=False)


    if removecorr is not None:
        df = remove_correlated(df, removecorr, general_cols)

    if dropna:
        df = df.dropna()

    return df


def report_dengue(path=DENGUE_PATH, dropna=False, usecols=None, usedefault=False, removecorr=None):
    df = load_dengue(path=path, dropna=dropna, usecols=usecols, usedefault=usedefault, removecorr=removecorr)
    create_report(df).save('dengue_report')


def load_pathology(path=PATHOLOGY_PATH, dropna=False, usecols=None, usedefault=False, removecorr=None):
    general_cols = ["_uid", "dateResult", "GenderID", "patient_age", "covid_confirmed"]

    if usedefault:

        FBC_features = ["EOS", "MONO", "BASO", "NEUT",
                        "RBC", "WBC", "MCHC", "MCV",
                        "LY", "HCT", "RDW", "HGB",
                        "MCH", "PLT", "MPV", "NRBCA"]

        usecols = general_cols + FBC_features

    if usecols and "dateResult" in usecols:
        df = pd.read_csv(path, usecols=usecols, low_memory=False, parse_dates=['dateResult'])
    else:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)

    if removecorr is not None:
        df = remove_correlated(df, removecorr, general_cols)

    if dropna:
        df = df.dropna()

    return df


def report_pathology(path=PATHOLOGY_PATH, dropna=False, usecols=None, usedefault=False, removecorr=None):
    df = load_pathology(path=path, dropna=dropna, usecols=usecols, usedefault=usedefault, removecorr=removecorr)
    create_report(df).save('pathology_report')


if __name__ == "__main__":
    #report_pathology(usedefault=True, dropna=True, removecorr=0.9)
    #report_dengue(usedefault=True, dropna=False)
    #df = load_dengue(usedefault=True)

    df = load_dengue(usedefault=True)

    mapping = {'Female': 1, 'Male': 2, True: 1, False: 0}
    df = df.replace({'gender': mapping, 'bleeding': mapping, 'shock': mapping,
                     'bleeding_gum': mapping, 'abdominal_pain': mapping, 'ascites': mapping,
                     'bleeding_mucosal': mapping, 'bleeding_skin': mapping})

    features = ["date", "age", "gender", "weight", "bleeding", "plt",
                "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

    for feat in features:
        df[feat] = df.groupby('study_no')[feat].ffill().bfill()

    df = df.loc[df['age'] <= 18]
    df = df.dropna()

    df = df.groupby(by="study_no", dropna=False).agg(
        date=pd.NamedAgg(column="date", aggfunc="last"),
        age=pd.NamedAgg(column="age", aggfunc="max"),
        gender=pd.NamedAgg(column="gender", aggfunc="first"),
        weight=pd.NamedAgg(column="weight", aggfunc=np.mean),
        bleeding=pd.NamedAgg(column="bleeding", aggfunc="max"),
        plt=pd.NamedAgg(column="plt", aggfunc="max"),
        shock=pd.NamedAgg(column="shock", aggfunc="max"),
        haematocrit_percent=pd.NamedAgg(column="haematocrit_percent", aggfunc="max"),
        bleeding_gum=pd.NamedAgg(column="bleeding_gum", aggfunc="max"),
        abdominal_pain=pd.NamedAgg(column="abdominal_pain", aggfunc="max"),
        ascites=pd.NamedAgg(column="ascites", aggfunc="max"),
        bleeding_mucosal=pd.NamedAgg(column="bleeding_mucosal", aggfunc="max"),
        bleeding_skin=pd.NamedAgg(column="bleeding_skin", aggfunc="max"),
        body_temperature=pd.NamedAgg(column="body_temperature", aggfunc=np.mean),
    ).dropna()

    df.to_csv('dengue.csv')
