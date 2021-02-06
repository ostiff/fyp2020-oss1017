import pandas as pd
import SimpSOM as sps
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

path = '../../../data/daily-profile.csv'

print("Loading dataset")

GENERAL_COLS = ["_uid", "dateResult", "GenderID", "patient_age", "covid_confirmed"]

BONE_features = ["GLOB","TP","CALC","CALCOR","ALP","PHOS","ALB"]

FBC_features = ["EOS", "MONO", "BASO", "NEUT",
                "RBC", "WBC", "MCHC", "MCV",
                "LY", "HCT", "RDW", "HGB",
                "MCH", "PLT", "MPV", "NRBCA"]

FBC_remove = ["WBC", "HGB", "HCT", "MCV"]   # remove because they have
                                            # correlations to other columns of over 0.9

panel_features = [item for item in FBC_features if item not in FBC_remove]

df = pd.read_csv(path, usecols=panel_features+GENERAL_COLS)
df = df.dropna()
df = df.drop_duplicates(subset='_uid', keep="first")
df = df.drop(columns=["_uid", "dateResult", "covid_confirmed"])

x = df.values #returns a numpy array
scaler = preprocessing.MinMaxScaler()
#scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)


net = sps.somNet(20, 20, x, PBC=True)
net.train(0.01, 20000)

for i in range(len(df.columns)):
    net.nodes_graph(colnum=i)
    plt.show()

net.diff_graph()
plt.show()

prj=np.array(net.project(x))
plt.scatter(prj.T[0],prj.T[1])
plt.show()

kmeans = KMeans(n_clusters=4, random_state=0).fit(prj)

plt.scatter(prj[:,0],prj[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()


