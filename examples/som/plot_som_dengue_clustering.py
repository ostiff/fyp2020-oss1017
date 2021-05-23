"""
SOM: Dengue dataset
===================

..TODO: Add description

Training attributes: `age`, `weight`, `plt`, `haematocrit_percent`,
`body_temperature`.

Attributes used in cluster comparison: `bleeding`, `shock`, `bleeding_gum`,
`abdominal_pain`, `ascites`, `bleeding_mucosal`, `bleeding_skin`.

"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from sklearn import preprocessing
from sklearn.cluster import KMeans
from tableone import TableOne

# Utils
from pkgname.utils.som_utils import diff_graph_hex, feature_maps, project_hex
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger
from pkgname.core.AE.autoencoder import set_seed
from pkgname.utils.print_utils import suppress_stderr, suppress_stdout
import pkgname.core.evaluation.dr_evaluation as dr_evaluation
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# Configuration
rcParams.update({'figure.autolayout': True})

nh_fn = ['gaussian', 'mexican_hat', 'bubble', 'triangle']
learning_rates = [5,2.5,1,0.5,0.25,0.1,0.08,0.05]
sig = [20,15,10,5,2.5,1,0.05]

grid_search = [[nh, s, lr] for nh in nh_fn
                 for s in sig
                 for lr in learning_rates]

start_idx = 4

for nh, s, lr in tqdm(grid_search[start_idx:]):
    with Logger('SOM_Dengue_grid_search') as logger, suppress_stdout(), suppress_stderr():
        try:
            N_CLUSTERS = 3
            SOM_X_SIZE = 53
            SOM_Y_SIZE = 13
            SOM_SIGMA = s
            SOM_lr = lr
            SOM_ACTIVATION_DIST = 'euclidean'
            SOM_NEIGHBOURHOOD = nh
            SEED = 0
            np.random.seed(SEED)
            set_seed(SEED)

            # %-----------
            # Load dataset
            # ------------

            features = ["dsource","date", "age", "gender", "weight", "bleeding", "plt",
                        "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                        "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

            df = load_dengue(usecols=['study_no']+features)

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

            df = IQR_rule(df, ['plt'])

            mapping = {'Female': 0, 'Male': 1}
            before_mapping = df.copy()

            df = df.replace({'gender': mapping})

            info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                       "bleeding_mucosal", "bleeding_skin", "gender"]
            data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

            info = df[info_feat]
            data = df[data_feat]
            print('len', len(data.index))
            scaler = preprocessing.StandardScaler()
            x = scaler.fit_transform(data.values)

            # ----------------------
            # Train SOM
            # ----------------------
            # Create SOM

            # with suppress_stderr() and suppress_stdout():
            som = MiniSom(SOM_X_SIZE, SOM_Y_SIZE, x.shape[1],
                topology='hexagonal',
                activation_distance=SOM_ACTIVATION_DIST,
                neighborhood_function=SOM_NEIGHBOURHOOD,
                sigma=SOM_SIGMA, learning_rate=SOM_lr,
                random_seed=SEED)

            # Train
            som.pca_weights_init(x)
            som.train_random(x, 1000000, verbose=True)

            projected_points = project_hex(som, x)

            logger.save_object(projected_points, "som_embedded")

            diff_graph_hex(som, show=False, printout=False)
            logger.add_plt(plt.gcf())
            plt.clf()

            feature_maps(som, feature_names=data_feat, cols=2, show=False, printout=False)
            logger.add_plt(plt.gcf())
            plt.clf()

            # Evaluate dimensionality reduction
            # Distance metrics
            res, fig = dr_evaluation.distance_metrics(x, projected_points, n_points=1000, method_name='SOM', verbose=True)
            logger.add_plt(plt.gcf(), ext='png')
            plt.clf()
            logger.add_parameters(res)

            # Density metrics
            res, fig = dr_evaluation.density_metrics(info, projected_points, ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                       "bleeding_mucosal", "bleeding_skin"], 'SOM')
            logger.add_plt(plt.gcf(), ext='png')
            plt.clf()
            logger.add_parameters(res)


            # ----------
            # Clustering
            # ----------

            colours = colours[:N_CLUSTERS]

            proj = project_hex(som, x)

            # Perform clustering on encoded inputs
            cluster = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(proj)

            labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

            scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster, cmap=ListedColormap(colours))
            plt.legend(handles=scatter.legend_elements()[0], labels=labels)
            logger.add_plt(plt.gcf())
            plt.clf()

            # %%
            #

            # ----------------
            # Cluster analysis
            # ----------------
            #
            # Table
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                before_mapping['cluster'] = cluster

            columns = info_feat+data_feat
            nonnormal = list(before_mapping[columns].select_dtypes(include='number').columns)
            categorical = list(set(columns).difference(set(nonnormal))) + ['dsource']
            columns = sorted(categorical) + sorted(nonnormal)

            rename = {'haematocrit_percent': 'hct',
                      'body_temperature': 'temperature'}

            table = TableOne(before_mapping, columns=columns, categorical=categorical, nonnormal=nonnormal,
                             groupby='cluster', rename=rename, missing=False)


            html = formatTable(table, colours, labels)
            logger.append_html(html.render())
            html


            # %%
            # These attributes were not used to train the model.

            fig, html = plotBox(data=info,
                                features=info_feat,
                                clusters=cluster,
                                colours=colours,
                                labels=labels,
                                title="Attributes not used in training",
                                )
            logger.append_html(html)
            fig

            #%%
            # The following attributes were used to train the model.

            fig, html = plotBox(data=data,
                                features=data_feat,
                                clusters=cluster,
                                colours=colours,
                                labels=labels,
                                title="Attributes used in training",
                                )
            logger.append_html(html)
            fig


            logger.save_parameters(
                {
                    'SEED': SEED,
                    'features': features,
                    'info_feat': info_feat,
                    'data_feat': data_feat,
                    'SOM_X_SIZE': SOM_X_SIZE,
                    'SOM_Y_SIZE': SOM_Y_SIZE,
                    'SOM_SIGMA': SOM_SIGMA,
                    'SOM_lr': SOM_lr,
                    'SOM_ACTIVATION_DIST': SOM_ACTIVATION_DIST,
                    'SOM_NEIGHBOURHOOD': SOM_NEIGHBOURHOOD
                }
            )

            logger.create_report()
        except Exception as e:
            logger.save_parameters(
                {
                    'EXCEPTION': str(e),
                    'SEED': SEED,
                    'features': features,
                    'info_feat': info_feat,
                    'data_feat': data_feat,
                    'SOM_X_SIZE': SOM_X_SIZE,
                    'SOM_Y_SIZE': SOM_Y_SIZE,
                    'SOM_SIGMA': SOM_SIGMA,
                    'SOM_lr': SOM_lr,
                    'SOM_ACTIVATION_DIST': SOM_ACTIVATION_DIST,
                    'SOM_NEIGHBOURHOOD': SOM_NEIGHBOURHOOD
                }
            )
            logger.create_report()
