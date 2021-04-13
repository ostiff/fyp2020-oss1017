import warnings
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plotBox(data, features, clusters, colours, title="Box plots" , path=None, disp=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        clusters = np.array(clusters)

        cols = 2
        rows = (len(features) + 1) // 2
        fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.05)

        for i, feat in enumerate(features):
            for j in range(len(colours)):
                fig.add_trace(
                    go.Box(
                        y=data[clusters == j][feat].values,
                        boxpoints='outliers', boxmean=True, name=f"Cluster {j}",
                        marker=dict(color=colours[j]),
                    ),
                    row=(i // cols) + 1, col=(i % cols) + 1
                )
            fig.update_yaxes(title_text=feat, row=(i // cols) + 1, col=(i % cols) + 1)

        fig.update_xaxes(showticklabels=False)
        fig.update_layout(height=477 * rows // 2, title_text=title, showlegend=False)

    if disp:
        fig.show()
    if path:
        fig.write_html(path)

    return fig