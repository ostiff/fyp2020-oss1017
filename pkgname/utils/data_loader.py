#TODO: include scaling?

import pandas as pd
from definitions import DENGUE_PATH, PATHOLOGY_PATH
from dataprep.eda import create_report

#TODO: Add a way of excluding columns
def remove_correlated(data, threshold):
    # Removed columns
    removed = set()

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


def load_dengue(path=DENGUE_PATH, dropna=True, usecols=None, usedefault=True, removecorr=None):
    if usecols is None and usedefault:
        general = ["study_no", "date", "age", "gender"]

        features = ["bleeding", "plt", "shock", "haematocrit_percent",
                        "bleeding_gum"]

        usecols = general + features

    df = pd.read_csv(path, usecols=usecols, low_memory=False, parse_dates=['date'])

    #TODO: don't remove certain columns
    if removecorr is not None:
        df = remove_correlated(df, removecorr)

    if dropna:
        df = df.dropna()

    return df


def report_dengue(path=DENGUE_PATH, dropna=True, usecols=None, usedefault=True, removecorr=None):
    df = load_dengue(path=path, dropna=dropna, usecols=usecols, usedefault=usedefault, removecorr=removecorr)
    create_report(df).save('dengue_report')


def load_pathology(path=PATHOLOGY_PATH, dropna=True, usecols=None, usedefault=True, removecorr=None):
    if usecols is None and usedefault:
        GENERAL_COLS = ["_uid", "dateResult", "GenderID", "patient_age", "covid_confirmed"]

        FBC_features = ["EOS", "MONO", "BASO", "NEUT",
                        "RBC", "WBC", "MCHC", "MCV",
                        "LY", "HCT", "RDW", "HGB",
                        "MCH", "PLT", "MPV", "NRBCA"]

        usecols = GENERAL_COLS + FBC_features #[item for item in FBC_features if item not in FBC_remove]

    df = pd.read_csv(path, usecols=usecols)

    #TODO: don't remove certain columns e.g. age/covid_confirmed
    if removecorr is not None:
        df = remove_correlated(df, removecorr)

    if dropna:
        df = df.dropna()

    return df


def report_pathology(path=PATHOLOGY_PATH, dropna=True, usecols=None, usedefault=True, removecorr=None):
    df = load_pathology(path=path, dropna=dropna, usecols=usecols, usedefault=usedefault, removecorr=removecorr)
    create_report(df).save('pathology_report')


if __name__ == "__main__":
    #print(load_pathology())
    #report_pathology(dropna=True,removecorr=0.9)

    #print(load_dengue(usedefault=True, dropna=False))
    report_dengue(dropna=True, removecorr=0.9)

