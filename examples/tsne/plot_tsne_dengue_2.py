"""
t-SNE: Dengue dataset 2
============================

Training attributes: `age`, `weight`, `plt`, `haematocrit_percent`,
`body_temperature`.

Attributes used in cluster comparison: `bleeding`, `shock`, `bleeding_gum`,
`abdominal_pain`, `ascites`, `bleeding_mucosal`, `bleeding_skin`, `gender`.

"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
# Libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tableone import TableOne

from pkgname.utils.data_loader import load_dengue, IQR_rule
from pkgname.utils.plot_utils import plotBox, formatTable, colours
from pkgname.utils.log_utils import Logger
from pkgname.core.AE.autoencoder import set_seed
import pkgname.core.evaluation.dr_evaluation as dr_evaluation
from pkgname.utils.print_utils import suppress_stdout, suppress_stderr

from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Agg')

perplexities = [5,10,15,20,25,30,35,40,45,50,100,200]
early_exaggerations = [5, 12, 20, 40]
learning_rates = [100,200,400]

grid_search = [[perp, exag, lr] for perp in perplexities
                 for exag in early_exaggerations
                 for lr in learning_rates]

start_idx = 123

for perp, exag, lr in tqdm(grid_search[start_idx:]):
    with Logger('TSNE_Dengue_grid_search', enable=True) as logger, suppress_stdout(), suppress_stderr():
        SEED = 0
        TSNE_n_components = 2
        TSNE_perplexity = perp
        TSNE_early_exaggeration = exag
        TSNE_learning_rate = lr
        DBSCAN_eps = 4
        DBSCAN_min_samples = 15

        np.random.seed(SEED)
        set_seed(SEED)

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
            age=pd.NamedAgg(column="age", aggfunc="min"),
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
        before_mapping = df.copy()

        df = df.replace({'gender': mapping})

        info_feat = ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                   "bleeding_mucosal", "bleeding_skin", "gender"]
        data_feat = ["age", "weight", "plt", "haematocrit_percent", "body_temperature"]

        info = df[info_feat]
        data = df[data_feat]

        scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(data.values)

        start_train = time.time()

        X_embedded = TSNE(n_components=TSNE_n_components,
                          perplexity=TSNE_perplexity,
                          early_exaggeration=TSNE_early_exaggeration,
                          learning_rate=TSNE_learning_rate,
                          random_state=SEED, n_jobs=-1).fit_transform(x)

        stop_train = time.time()
        train_time = stop_train - start_train

        start_eval = time.time()
        logger.save_object(X_embedded, "X_embedded")
        plt.scatter(X_embedded[:,0], X_embedded[:,1], c=info['shock'])
        plt.clf()

        clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(X_embedded)
        outliers = -1 in clustering.labels_
        clusters = [x+1 for x in clustering.labels_] if outliers else clustering.labels_


        # Evaluate dimensionality reduction
        # Distance metrics
        res, fig = dr_evaluation.distance_metrics(x, X_embedded, n_points=1000, method_name='t-SNE', verbose=True)
        logger.add_plt(plt.gcf(), ext='png')
        plt.clf()
        logger.add_parameters(res)

        # Density metrics
        res, fig = dr_evaluation.density_metrics(info, X_embedded, ["shock", "bleeding", "bleeding_gum", "abdominal_pain", "ascites",
                   "bleeding_mucosal", "bleeding_skin"], 't-SNE')
        logger.add_plt(plt.gcf(), ext='png')
        plt.clf()
        logger.add_parameters(res)


        # %%
        # Plotting
        # --------

        N_CLUSTERS = len(set(clusters))
        print('n_clusters = ', N_CLUSTERS)
        colours = colours[:N_CLUSTERS]

        scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=clusters, cmap=ListedColormap(colours))

        if outliers:
            labels = ["Outliers"] + [f"Cluster {i}" for i in range(N_CLUSTERS-1)]
        else:
            labels= [f"Cluster {i}" for i in range(N_CLUSTERS)]

        plt.legend(handles=scatter.legend_elements()[0], labels=labels)
        plt.title('t-SNE + DBSCAN')
        logger.add_plt(plt.gcf())
        plt.clf()


        # %%
        # Cluster analysis
        # ----------------
        #
        # Table

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            before_mapping['cluster'] = clusters

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
                            clusters=clusters,
                            colours=colours,
                            labels=labels,
                            title="Attributes not used in training",
                            #path="a.html"
                            )
        logger.append_html(html)
        fig

        #%%
        # The following attributes were used to train the model.

        fig, html = plotBox(data=data,
                            features=data_feat,
                            clusters=clusters,
                            colours=colours,
                            labels=labels,
                            title="Attributes used in training",
                            #path="b.html"
                            )
        logger.append_html(html)
        fig

        end_eval = time.time()
        eval_time = end_eval - start_eval

        # %%
        # Logging
        # -------

        # Log parameters
        logger.save_parameters(
            {
                'SEED': SEED,
                'TSNE_n_components': TSNE_n_components,
                'TSNE_perplexity': TSNE_perplexity,
                'TSNE_early_exaggeration': TSNE_early_exaggeration,
                'TSNE_learning_rate': TSNE_learning_rate,
                'DBSCAN_eps': DBSCAN_eps,
                'DBSCAN_min_samples': DBSCAN_min_samples,
                'features': features,
                'info_feat': info_feat,
                'data_feat': data_feat,
                'train_time': train_time,
                'eval_time': eval_time
            }
        )

        logger.create_report()
