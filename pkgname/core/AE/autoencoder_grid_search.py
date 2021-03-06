"""
Simple autoencoder for dengue clustering
========================================

Training attributes: `age`, `weight`, `plt`, `haematocrit_percent`,
`body_temperature`.

Attributes used in cluster comparison: `bleeding`, `shock`, `bleeding_gum`,
`abdominal_pain`, `ascites`, `bleeding_mucosal`, `bleeding_skin`.

"""
import logging
logging.disable(logging.CRITICAL)


import os
import sys
sys.path.insert(0, os.path.abspath('.'))
# Libraries
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tableone import TableOne

from pkgname.core.AE.autoencoder import Autoencoder, train_autoencoder, plot_autoencoder_loss, get_device, set_seed
from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger
import pkgname.core.evaluation.dr_evaluation as dr_evaluation

from pkgname.utils.print_utils import suppress_stdout, suppress_stderr

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

layers_list = [[],[5],[4],[3],[5,4],[5,3],[4,3],[5,4,3]]
learning_rates = [0.1, 0.05, 0.01, 0.005,0.001,0.0005,0.0001,0.00005,0.00001]
epochs_list = [10, 30, 50, 100, 150, 250, 350, 500]
b_size_list = [16, 32]

grid_search = [[lay, epochs, lr, b_size] for lay in layers_list
                 for epochs in epochs_list
                 for lr in learning_rates
                 for b_size in b_size_list]

start_idx = 907

for lay, epochs, lr, b_size in tqdm(grid_search[start_idx:]):
    with Logger('AE_Dengue_grid_search_sigmoid', enable=True) as logger, suppress_stdout(), suppress_stderr():
        try:
            SEED = 0
            N_CLUSTERS = 3

            # Set seed
            set_seed(SEED)

            # Get device
            device = get_device(usegpu=False)

            num_epochs = epochs
            learning_rate = lr
            batch_size = b_size
            latent_dim = 2
            layers=lay

            features = ["dsource","date", "age", "gender", "weight", "bleeding", "plt",
                        "shock", "haematocrit_percent", "bleeding_gum", "abdominal_pain",
                        "ascites", "bleeding_mucosal", "bleeding_skin", "body_temperature"]

            info_feat = ["dsource", "shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                       "bleeding_mucosal", "bleeding_skin", "gender"]
            data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]


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
                body_temperature=pd.NamedAgg(column="body_temperature", aggfunc="max"),
            ).dropna()

            df = IQR_rule(df, ['plt'])

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
            train_scaled_ = train_scaled.copy()
            loader_train_no_shuffle = DataLoader(train_scaled_, batch_size, shuffle=False)

            loader_train = DataLoader(train_scaled, batch_size, shuffle=True)
            loader_test = DataLoader(test_scaled, batch_size, shuffle=False)

            # Additional parameters
            input_size = len(data_feat)
            model = Autoencoder(input_size=input_size,
                                layers=layers,
                                latent_dim=latent_dim,
                                device=device).to(device)

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train
            losses = train_autoencoder(model, optimizer, loader_train, loader_test, num_epochs,
                                       # animation_data=loader_train_no_shuffle,
                                       # animation_colour=[train_info['shock'],
                                       #                   train_info['bleeding'],
                                       #                   train_info['ascites'],
                                       #                   train_info['abdominal_pain'],
                                       #                   train_info['bleeding_mucosal'],
                                       #                   train_info['bleeding_gum'],
                                       #                   train_info['bleeding_skin'],
                                       #                   train_info['gender']],
                                       # animation_labels=['Shock', 'Bleeding', 'Ascites', 'Abdominal pain',
                                       #                   'Bleeding mucosal', 'Bleeding gum',
                                       #                   'Bleeding skin', 'Gender'],
                                       # animation_path='animation.gif'
                                       )

            # Save model
            logger.save_object(model)


            # %%
            #

            # Plot losses
            plot = plot_autoencoder_loss(losses, show=False, printout=False)
            logger.add_plt(plot)
            plt.clf()

            # %%
            #

            colours = colours[:N_CLUSTERS]

            # Encode test set and plot in 2D (assumes latent_dim = 2)

            encoded_test = model.encode_inputs(loader_test)
            encoded_train = model.encode_inputs(loader_train_no_shuffle)


            plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=test_info['shock'])
            plt.title('AE shock in latent space (testing data)')
            logger.add_plt(plt.gcf())
            plt.clf()

            # %%
            #

            plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=train_info['shock'])
            plt.title('AE shock in latent space (training data)')
            logger.add_plt(plt.gcf())
            plt.clf()

            # %%
            #

            # Evaluate dimensionality reduction
            # Distance metrics
            res, fig = dr_evaluation.distance_metrics(train_scaled, encoded_train, n_points=1000, method_name='AE', verbose=True)
            logger.add_plt(plt.gcf(), ext='png')
            plt.clf()
            logger.add_parameters(res)

            # %%
            #

            # Density metrics
            res, fig = dr_evaluation.density_metrics(train_info, encoded_train, ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                       "bleeding_mucosal", "bleeding_skin"], 'AE')
            logger.add_plt(plt.gcf(), ext='png')
            plt.clf()
            logger.add_parameters(res)

            # %%
            #
            clusters_test = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(encoded_test)
            clusters_train = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(encoded_train)

            labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]

            scatter = plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=clusters_test, cmap=ListedColormap(colours))
            plt.legend(handles=scatter.legend_elements()[0], labels=labels)
            plt.title('AE k-means (testing data)')
            logger.add_plt(plt.gcf())
            plt.clf()

            # %%
            #

            scatter = plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=clusters_train, cmap=ListedColormap(colours))
            plt.legend(handles=scatter.legend_elements()[0], labels=labels)
            plt.title('AE k-means (training data)')
            logger.add_plt(plt.gcf())
            plt.clf()

            # %%
            #


            # Table
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mapping = {0: 'Female', 1: 'Male'}
                table_df = test.replace({'gender': mapping})

                table_df['cluster'] = clusters_test

            columns = info_feat+data_feat
            nonnormal = list(table_df[columns].select_dtypes(include='number').columns)
            categorical = list(set(columns).difference(set(nonnormal)))
            columns = sorted(categorical) + sorted(nonnormal)

            rename = {'haematocrit_percent': 'hct',
                      'body_temperature': 'temperature'}

            table = TableOne(table_df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                             groupby='cluster', rename=rename, missing=False)

            html = formatTable(table, colours, labels)
            logger.append_html(html.render())
            html

            # %%
            #

            fig, html = plotBox(data=test_info,
                                features=info_feat,
                                clusters=clusters_test,
                                colours=colours,
                                title="Attributes not used in training",
                                )
            logger.append_html(html)
            fig

            # %%
            #

            fig, html = plotBox(data=test_data,
                                features=data_feat,
                                clusters=clusters_test,
                                colours=colours,
                                title="Attributes used in training",
                                )
            logger.append_html(html)
            fig

            # %%
            # Logging
            # -------

            # Log parameters
            logger.save_parameters(
                {
                    'SEED': SEED,
                    'N_CLUSTERS': N_CLUSTERS,
                    'device': str(device),
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'latent_dim': latent_dim,
                    'input_size':input_size,
                    'layers':layers,
                    'features': features,
                    'info_feat': info_feat,
                    'data_feat': data_feat
                }
            )

            logger.create_report()
            plt.close('all')
        except Exception as e:
            logger.save_parameters(
                {
                    'EXCEPTION': str(e),
                    'SEED': SEED,
                    'N_CLUSTERS': N_CLUSTERS,
                    'device': str(device),
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'latent_dim': latent_dim,
                    'input_size': input_size,
                    'layers': layers,
                    'features': features,
                    'info_feat': info_feat,
                    'data_feat': data_feat
                }
            )

            logger.create_report()
            plt.close('all')