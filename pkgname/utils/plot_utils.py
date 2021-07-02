import itertools
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from plotly.subplots import make_subplots
from plotly.io import to_html
import plotly.graph_objects as go
from scipy import linalg
from seaborn import color_palette
import matplotlib.colors as mc
import colorsys
import matplotlib.pyplot as plt

colours = color_palette(as_cmap=True) + color_palette('pastel',as_cmap=True) + ["#D8BFD8","#CD919E","#7F7F7F","#FFEFD5","#CD96CD","#9400D3","#F5F5DC","#FF9912","#FAFAFA","#969696","#E066FF","#FAEBD7","#FFEFDB","#9BCD9B","#FA8072","#EECBAD","#CD2626","#8470FF","#CD950C","#EE82EE","#EE3A8C","#8B2323","#FF7F50","#CDB79E","#BF3EFF","#C1FFC1","#556B2F","#8B7B8B","#668B8B","#E8E8E8","#CD9B9B","#050505","#CAFF70","#F7F7F7","#FFD700","#FF82AB","#009ACD","#8B864E","#EE7600","#EEA2AD","#FFD39B","#949494","#FDF5E6","#8968CD","#EEC591","#6CA6CD","#71C671","#FFC1C1","#838B83","#555555","#C4C4C4","#CD1076","#00FFFF","#EED5B7","#8B5A00","#8B8970","#8B8386","#4A708B","#8FBC8F","#EEE9BF","#E3A869","#FFAEB9","#FF34B3","#CD69C9","#32CD32","#9C9C9C","#A0522D","#D3D3D3","#8C8C8C","#FFF68F","#B23AEE","#BCEE68","#B0171F","#8B1C62","#EE4000","#8B3E2F","#98F5FF","#BDBDBD","#FFA54F","#FFC0CB","#B8860B","#212121","#FFFAF0","#C67171","#EEB422","#BCD2EE","#5D478B","#FFF5EE","#CD853F","#778899","#D4D4D4","#00688B","#FFA500","#FFBBFF","#00EEEE","#EE30A7","#ABABAB","#008080","#CD8500","#171717"]
pastel = color_palette('pastel',as_cmap=True) + color_palette(as_cmap=True)

def adjust_lightness(colour, multiplier=0.75):
    try:
        c = mc.cnames[colour]
    except:
        c = colour

    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, multiplier * c[1])), c[2])

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
                if data[feat].dtype == bool:
                    percent_true = data[clusters == j][feat].sum() / len(data[clusters == j]) * 100
                    fig.add_trace(
                        go.Bar(
                            x=['True %'], y=[percent_true],
                            name=labels[j] + ' %',
                            marker=dict(color=colours[j],
                                        opacity=0.6),
                        ),
                        row=(i // cols) + 1, col=(i % cols) + 1
                    )
                    fig.update_yaxes(title_text=feat, row=(i // cols) + 1, col=(i % cols) + 1, range=[0,100])
                else:
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


def formatTable(table, colours, labels, resize_header=True):
    table_df = table.tableone

    # Drop 'Group by' index level created by TableOne
    table_df.columns = table_df.columns.droplevel()

    # Rename groupby columns
    table_df = table_df.rename(columns=dict(zip([str(i) for i in range(len(labels))], labels)))

    th_format = {}
    for i, l in enumerate(labels):
        c = colours[i] if i < len(colours) else '#000000'

        th_format[l] = [dict(selector='th', props=[('color', c)])]

    table = table_df.style.set_table_styles(th_format, overwrite=False)

    if resize_header:
        styles = [
            dict(selector="th.col_heading", props=[("font-size", "130%"),
                                                   ("text-align", "right")]),
        ]

        table = (table.set_table_styles(styles))

    return table

  
def format_table_bootstrap(table, colours, labels):
    table = formatTable(table, colours, labels, resize_header=False)
    table.set_table_attributes('class="table table-bordered table-striped table-condensed mb-none"')

    return table.render()


def plot_results(X, Y_, means, covariances, colours, labels):
    plt.figure()
    splot = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y_, palette=colours,
                            linewidth=0, s=10)

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, colours)):
        color=colours[color]
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    handles, _ = splot.get_legend_handles_labels()
    splot.legend(handles, labels, loc='lower right', borderpad=0.2, labelspacing=0.2)
