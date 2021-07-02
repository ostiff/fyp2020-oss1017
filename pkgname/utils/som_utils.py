from minisom import MiniSom

import pandas as pd
import numpy as np
import os
import math

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D



def project_hex(som, data):
    """Project the datapoints of a given array to the 2D space of a hex SOM.

    :param MiniSom som: MiniSom object.
    :param np.array data: Array containing datapoints to be mapped.
    :return np.array: array containing
    """

    proj = np.empty([data.shape[0],2])
    for i, x in enumerate(data):
        # getting the winner
        w = som.winner(x)
        # place a marker on the winning position for the sample xx
        wx, wy = som.convert_map_to_euclidean(w)
        wy = wy * 2 / np.sqrt(3) * 3 / 4
        proj[i] = [wx, wy]

    return proj


def create_hex_plt(som, ax, data):
    """Create 2D hex plot in a given axis using given data.

    :param MiniSom som: MiniSom object.
    :param matplotlib.axes.Axes ax: Matplotlib ax to plot on.
    :param data: Data to display in hex plot.
    :return:
    """
    xx, yy = som.get_euclidean_coordinates()

    ax.set_aspect('equal')

    # iteratively add hexagons
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
            hex = RegularPolygon((xx[(i, j)], wy),
                                 numVertices=6,
                                 radius=.98 / np.sqrt(3),
                                 facecolor=cm.viridis(data[i, j]),
                                 edgecolor='white')
            ax.add_patch(hex)

    ax.plot()


def project_labels(som, data, labels, label_names=None, show=False, printout=True, disp_axes=False, dpi=300,
                   fname='./node_differences.jpg'):

    proj = project_hex(som, data)
    umatrix = som.distance_map()

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    create_hex_plt(som, ax, umatrix)

    num_labels = len(set(labels))
    if label_names is None:
        label_names = range(num_labels)
    markers = [marker for marker, val in Line2D.markers.items() if val != 'nothing' and val != 'pixel'][:num_labels]
    colors = ['C%s' % i for i in range(num_labels)]

    for cnt, (wx, wy) in enumerate(proj):
        plt.plot(wx, wy,
                 markers[labels[cnt] - 1],
                 markerfacecolor='None',
                 markeredgecolor=colors[labels[cnt] - 1],
                 markersize=12,
                 markeredgewidth=2)

    if disp_axes:
        xrange = np.arange(umatrix.shape[0])
        yrange = np.arange(umatrix.shape[1])
        plt.xticks(xrange - 0.5, xrange)
        plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)
        pad = 0.1
    else:
        plt.axis('off')
        pad = 0

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=pad)
    colorbar.ColorbarBase(ax_cb, cmap=cm.viridis, orientation='vertical')

    plt.suptitle('Distance from neurons in the neighbourhood', fontsize=24)
    plt.gcf().add_axes(ax_cb)

    legend_elements = [Line2D([0], [0], marker=markers[i], color=colors[i], label=name,
                              markerfacecolor='w', markersize=14, linestyle='None',
                              markeredgewidth=2) for i, name in enumerate(label_names) ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
              borderaxespad=0., ncol=min(60//len(max(list(map(str,label_names)), key=len)), num_labels), fontsize=14)

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()



def diff_graph_hex(som, show=False, printout=True, disp_axes=False, dpi=300,
                   fname='./node_differences.jpg'):
    """Plot a 2D hex map showing weight differences.

    :param MiniSom som: MiniSom object.
    :param bool show: Display the plot.
    :param bool printout: Save the plot to a file.
    :param bool disp_axes: Display plot axes.
    :param int dpi: Figure DPI.
    :param str fname: Path and file name including extension.
    :return:
    """

    umatrix = som.distance_map()

    f = plt.figure(figsize=(15, 4))
    ax = f.add_subplot(111)

    create_hex_plt(som, ax, umatrix)

    if disp_axes:
        xrange = np.arange(umatrix.shape[0])
        yrange = np.arange(umatrix.shape[1])
        plt.xticks(xrange-0.5, xrange)
        plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)
        pad = 0.1
    else:
        plt.axis('off')
        pad = 0

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=pad)
    colorbar.ColorbarBase(ax_cb, cmap=cm.viridis, orientation='vertical')

    plt.suptitle('Distance from neurons in the neighbourhood', fontsize=24)
    plt.gcf().add_axes(ax_cb)

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()

    return f


def number_samples(som, data, show=False, printout=True, disp_axes=False, dpi=300,
                   fname='./node_differences.jpg'):
    umatrix = som.distance_map()

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)

    create_hex_plt(som, ax, umatrix)
    data = project_hex(som, data)
    for i, (x, y) in enumerate(data):
        plt.text(x, y, str(i + 1), color="red", fontsize=15)

    if disp_axes:
        xrange = np.arange(umatrix.shape[0])
        yrange = np.arange(umatrix.shape[1])
        plt.xticks(xrange-0.5, xrange)
        plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)
        pad = 0.1
    else:
        plt.axis('off')
        pad = 0

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=pad)
    colorbar.ColorbarBase(ax_cb, cmap=cm.viridis, orientation='vertical')

    plt.suptitle('Distance from neurons in the neighbourhood', fontsize=24)
    plt.gcf().add_axes(ax_cb)

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()


def feature_map(som, colnum=0, show=False, printout=True, disp_axes=False, dpi=300,
                   fname='./node_differences.jpg'):

    """Plot a 2D map with hexagonal nodes for a feature specified by colnum.

    :param MiniSom som: MiniSom object.
    :param int colnum: The index of the weight that will be shown as colormap.
    :param bool show: Display the plot.
    :param bool printout: Save the plot to a file.
    :param bool disp_axes: Display plot axes.
    :param int dpi: Figure DPI.
    :param str fname: Path and file name including extension.
    :return:
    """

    weights = som.get_weights()[:, :, colnum]

    f = plt.figure(figsize=(15, 4))
    ax = f.add_subplot(111)

    create_hex_plt(som, ax, weights)

    plt.suptitle("Node Grid for Feature %s" % colnum, fontsize=24)
    plt.tight_layout()

    if not disp_axes:
        ax.axis('off')

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()


def feature_maps(som, feature_names=None, cols=1, show=False, printout=True, disp_axes=False, dpi=300,
                   fname='./feature_maps.jpg'):
    """Plot a 2D map with hexagonal nodes for each feature in the dataset.

    :param MiniSom som: MiniSom object.
    :param  feature_names: List of feature names.
    :param int cols: Number of subplot columns.
    :param bool show: Display the plot.
    :param bool printout: Save the plot to a file.
    :param bool disp_axes: Display plot axes.
    :param int dpi: Figure DPI.
    :param str fname: Path and file name including extension.
    :return:
    """

    weights = som.get_weights()
    n_features = weights.shape[2]
    cols = min(n_features, cols)
    rows = math.ceil(n_features / cols)

    # Create figure
    f, axes = plt.subplots(rows, cols, figsize=(2*cols,2*rows))
    axes = axes.flatten()

    # Plot feature planes
    for i, f in enumerate(range(n_features)):
        create_hex_plt(som, axes[i], weights[:, :, f])
        title = 'Feature %s' % f if feature_names is None else feature_names[i]
        axes[i].set(title=title, aspect='equal',
            yticks=[], xticks=[])
        if not disp_axes:
            axes[i].axis('off')

    # Hide axes for unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    # Set axes
    if feature_names is None:
        plt.suptitle("Feature #i map")
    else:
        plt.suptitle("Feature maps")
    plt.tight_layout()

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()