from flask import Flask, request, jsonify, Response

import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch import FloatTensor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tableone import TableOne

from pkgname.core.VAE.vae import get_device, set_seed
from pkgname.utils.data_loader import load_dengue
from pkgname.utils.plot_utils import plotBox, formatTable
from pkgname.utils.log_utils import Logger
from definitions import ROOT_DIR


app = Flask(__name__)

@app.route('/get_data', methods=['GET'])
def get_data():
    x = encoded_test[:,0].tolist()
    y = encoded_test[:,1].tolist()
    resp = {
        'x': x,
        'y': y
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def makeTable(idx):
    c = [1 if (i in idx) else 0 for i in range(len(encoded_test))]
    table_df['cluster'] = c
    table = TableOne(table_df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                     groupby='cluster', rename=rename, missing=False, overall=False)
    html = formatTable(table, ["#1f77b4", "#ff7f0e"], ["Not selected", "Selected"])
    return html.render()


@app.route('/get_k_nearest', methods=['GET'])
def get_k_nearest():
    id = int(request.args.get('id'))
    k = int(request.args.get('k'))
    idx = tree.query([encoded_test[id]], k=k, return_distance=False)

    resp = {
        'idx': idx.tolist(),
        'table': makeTable(idx)
    }

    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/enc_patient', methods=['GET'])
def enc_patient():
    k = int(request.args.get('k'))
    age = float(request.args.get('age'))
    weight = float(request.args.get('weight'))
    plt = float(request.args.get('plt'))
    hct = float(request.args.get('hct'))
    b_temp = float(request.args.get('b_temp'))

    scaled = scaler.transform([[age, weight, plt, hct, b_temp]])
    inp = DataLoader(scaled, 1, shuffle=False)
    enc = model.encode_inputs(inp)
    idx = tree.query(enc, k=k, return_distance=False)

    resp = {
        'idx': idx.tolist(),
        'table': makeTable(idx)
    }

    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":

    # --------------
    # Load data
    # --------------

    SEED = 0
    batch_size = 16
    MODEL_PATH = os.path.join(ROOT_DIR, 'examples', 'vae', 'model')
    LEAF_SIZE = 40

    # Set seed
    set_seed(SEED)

    # Get device
    device = get_device(False)

    features = ["dsource", "date", "age", "gender", "weight", "bleeding", "plt",
                "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

    df = load_dengue(usecols=['study_no'] + features)

    for feat in features:
        df[feat] = df.groupby('study_no')[feat].ffill().bfill()

    df = df.loc[df['age'] <= 18]
    df = df.dropna()

    df = df.groupby(by="study_no", dropna=False).agg(
        dsource=pd.NamedAgg(column="dsource", aggfunc="last"),
        date=pd.NamedAgg(column="date", aggfunc="last"),
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

    mapping = {'Female': 0, 'Male': 1}
    df = df.replace({'gender': mapping})

    info_feat = ["dsource", "shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                 "bleeding_mucosal", "bleeding_skin", "gender"]
    data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

    train, test = train_test_split(df, test_size=0.2, random_state=SEED)

    train_data = train[data_feat]
    test_data = test[data_feat]
    train_info = train[info_feat]
    test_info = test[info_feat]

    scaler = preprocessing.MinMaxScaler().fit(train_data)

    train_scaled = scaler.transform(train_data.to_numpy())
    test_scaled = scaler.transform(test_data.to_numpy())

    loader_train = DataLoader(train_scaled, batch_size, shuffle=True)
    loader_test = DataLoader(test_scaled, batch_size, shuffle=False)

    # ------------------
    # Create tree
    # ------------------

    model = pickle.load(open(MODEL_PATH, 'rb'))
    encoded_test = model.encode_inputs(loader_test)
    tree = KDTree(encoded_test, leaf_size=LEAF_SIZE)

    # --------------
    # Table setup
    # --------------

    mapping = {0: 'Female', 1: 'Male'}
    table_df = test.replace({'gender': mapping})
    columns = (info_feat + data_feat)
    columns.remove("dsource")
    nonnormal = list(table_df[columns].select_dtypes(include='number').columns)
    categorical = list(set(columns).difference(set(nonnormal)))
    columns = sorted(categorical) + sorted(nonnormal)
    rename = {'haematocrit_percent': 'hct',
              'body_temperature': 'temperature'}

    app.run()

