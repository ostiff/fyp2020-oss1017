from minisom import MiniSom

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar


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


def diff_graph_hex(som, show=False, printout=True, axis=False, dpi=300,
                   fname='./node_differences.jpg'):
    """Plot a 2D hex map showing weight differences.

    :param MiniSom som: MiniSom object.
    :param bool show: Display the plot.
    :param bool printout: Save the plot to a file.
    :param bool axis: Display plot axes.
    :param int dpi: Figure DPI.
    :param str fname: Path and file name including extension.
    :return:
    """

    xx, yy = som.get_euclidean_coordinates()
    umatrix = som.distance_map()
    weights = som.get_weights()

    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)

    ax.set_aspect('equal')

    # iteratively add hexagons
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
            hex = RegularPolygon((xx[(i, j)], wy),
                                 numVertices=6,
                                 radius=.98 / np.sqrt(3),
                                 facecolor=cm.viridis(umatrix[i, j]),
                                 edgecolor='white')
            ax.add_patch(hex)

    plt.plot()

    if axis:
        xrange = np.arange(weights.shape[0])
        yrange = np.arange(weights.shape[1])
        plt.xticks(xrange-0.5, xrange)
        plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)
        pad = 0.1
    else:
        plt.axis('off')
        pad = 0

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=pad)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.viridis,
                                orientation='vertical')
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('Distance from neurons in the neighbourhood',
                      rotation=270, fontsize=16)
    plt.gcf().add_axes(ax_cb)

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
