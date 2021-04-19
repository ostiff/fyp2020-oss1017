import warnings
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from plotly.io import to_html
import plotly.graph_objects as go

def plotBox(data, features, clusters, colours, labels=None, title="Box plots" , path=None, disp=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        clusters = np.array(clusters)

        if not labels:
            labels = [f"Cluster {i}" for i in range(len(colours))]

        cols = 2
        rows = (len(features) + 1) // 2
        fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.05)

        for i, feat in enumerate(features):
            for j in range(len(colours)):
                fig.add_trace(
                    go.Box(
                        y=data[clusters == j][feat].values,
                        boxpoints='outliers', boxmean=True, name=labels[j],
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

    return fig, to_html(fig)


def formatTable(table, colours, labels):
    table_df = table.tableone

    # Drop 'Group by' index level created by TableOne
    table_df.columns = table_df.columns.droplevel()

    # Rename groupby columns
    table_df = table_df.rename(columns=dict(zip([str(i) for i in range(len(labels))], labels)))

    styles = [
        dict(selector="th.col_heading", props=[("font-size", "130%"),
                                               ("text-align", "right")]),
    ]

    html = (table_df.style.set_table_styles(styles))

    th_format = {}
    for i, l in enumerate(labels):
        th_format[l] = [dict(selector='th', props=[('color', colours[i])])]

    html = html.set_table_styles(th_format, overwrite=False)

    return html