"""
Auto-encoder interactive example (Code)
=======================================

Interactive example to visualise dengue data encoded by an auto encoder.

Features:

    - Get information about the k points closest to the selected one.
    - Input data corresponding to an unseen patient to get information about the k
      patients which are closest in the latent space.


With the virtual environment active:
``$ python examples/vae/vae_kdtree_server.py``

The server will be started locally on: http://127.0.0.1:5000/

The example can be accessed on http://127.0.0.1:5000/ or by opening
``examples/vae/templates/vae_kd_tree_client.html`` in a browser.
"""


from flask import Flask, request, jsonify, render_template

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KDTree
from tableone import TableOne

from pkgname.core.AE.autoencoder import get_device, set_seed
from pkgname.utils.data_loader import load_dengue
from pkgname.utils.plot_utils import formatTable
from definitions import ROOT_DIR


app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, 'examples', 'autoencoder', 'templates'))

# ------------
# Render Pages
# ------------


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/knn')
def knn():
    return render_template('ae_kd_tree_client.html')

@app.route('/trace')
def trace():
    return render_template('ae_patient_trace.html')


# --------
# API
# --------

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
                     groupby='cluster', rename=rename, missing=False, overall=False,
                     pval=True)
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


@app.route('/get_trace', methods=['GET'])
def get_trace():
    study_no = request.args.get('study_no', before_fill['study_no'].sample().values[0])
    patient_data = before_fill.loc[before_fill['study_no'] == study_no]

    if patient_data.empty:
        return "study_no not found", 400

    scaled = scaler.transform(patient_data[data_feat].to_numpy())
    inp = DataLoader(scaled, 1, shuffle=False)
    enc = model.encode_inputs(inp)
    patient_data['x'] = enc[:,0]
    patient_data['y'] = enc[:,1]
    patient_data = patient_data.drop_duplicates(subset=['x', 'y'])

    data = patient_data.sort_values('date', ignore_index=True)

    resp = {
        'study_no': study_no,
        'x': data['x'].tolist(),
        'y': data['y'].tolist(),
        'date': data['date'].tolist()
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
    MODEL_PATH = os.path.join(ROOT_DIR, 'examples', 'autoencoder', 'model')
    LEAF_SIZE = 40

    # Set seed
    set_seed(SEED)

    # Get device
    device = get_device(False)

    features = ["dsource", "date", "age", "gender", "weight", "bleeding", "plt",
                "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]
    info_feat = ["dsource", "shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                 "bleeding_mucosal", "bleeding_skin", "gender"]
    data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

    before_fill = load_dengue(usecols=['study_no'] + features)
    before_fill = before_fill.loc[before_fill['age'] <= 18]
    before_fill = before_fill.loc[before_fill['dsource'] != 'md']
    before_fill = before_fill.loc[before_fill['plt'] < 5000]

    df = before_fill.copy()

    before_fill = before_fill.dropna(subset=data_feat + ['date'])

    for feat in features:
        df[feat] = before_fill.groupby('study_no')[feat].ffill().bfill()

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

