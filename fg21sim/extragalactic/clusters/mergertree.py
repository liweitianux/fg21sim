# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license

"""
Merger tree that represents the merging history of a cluster using
the binary tree data structure.
"""

import os
import pickle
import logging

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


logger = logging.getLogger(__name__)


class MergerTree:
    """
    A binary tree that represents the cluster merging history.

    Description
    -----------
                   Merged (M0, z0, age0)
                   ~~~~~~~~~~~~~~~~~~~~~
                   /                   \
        Main (M1, z1, age1)    Sub (M2, z2, age2)
        ~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~
    * "Merged" is the merged cluster from "Main" and "Sub" at redshift
      z1 (=z2) or cosmic time age1 (=age2).
      M0 = M1 + M2, M1 > M2
    * If "Sub" is missing, then this is an accretion event (not a merger).

    Parameters
    ----------
    data : dict
        Data (e.g., mass, redshift, age) associated with this tree node.
    main, sub : `~MergerTree`
        Links to the main and sub (optional) clusters between which the
        merger happens.
        The ``sub`` cluster may be missing, which is regarded as an
        accretion event rather than a merger.
    """
    def __init__(self, data, main=None, sub=None):
        self.data = data
        self.main = main
        self.sub = sub


def save_mtree(mtree, outfile, clobber=False):
    """
    Pickle the merger tree data and save to file.
    """
    if os.path.exists(outfile):
        if clobber:
            os.remove(outfile)
            logger.warning("Removed existing file: {0}".format(outfile))
        else:
            raise OSError("Output file already exists: {0}".format(outfile))
    pickle.dump(mtree, open(outfile, "wb"))
    logger.info("Saved merger tree to file: {0}".format(outfile))


def read_mtree(infile):
    mtree = pickle.load(open(infile, "wb"))
    logger.info("Loaded merger tree from file: {0}".format(infile))
    return mtree


def show_mtree(mtree):
    """
    Trace the main cluster and show its formation history.
    """
    def _show_event(main, sub=None, parent=None):
        z = main.data["z"]
        age = main.data["age"]
        mass = main.data["mass"]
        info = "[z=%.3f/t=%.2f]" % (z, age)
        if sub is None:
            # Accretion
            info += " %.3e" % mass
            if parent is not None:
                dM = parent.data["mass"] - mass
                info += "    (dM=%.2e)" % dM
        else:
            # Merger
            Msub = sub.data["mass"]
            Rmass = mass / Msub
            info += " %.3e <> %.3e (Rm=%.1f)" % (mass, Msub, Rmass)
        return info

    i = 0
    info = _show_event(main=mtree)
    print("%2d %s" % (i, info))
    while mtree and mtree.main:
        i += 1
        info = _show_event(main=mtree.main, sub=mtree.sub, parent=mtree)
        print("%2d %s" % (i, info))
        mtree = mtree.main


def plot_mtree(mtree, outfile, figsize=(12, 8)):
    """
    Plot the cluster merger tree.

    TODO/XXX: This function needs significant speed optimization!

    Parameters
    ----------
    mtree : `~MergerTree`
        The merger tree to be plotted
    outfile : str
        Output filename to save the plotted figure
    figsize : tuple
        The (width, height) of the plotting figure
    """
    def _plot(tree, ax):
        if tree is None:
            return
        if tree.main is None:
            # Only plot a point for current tree node
            x = [tree.data["age"]]
            y = [tree.data["mass"]]
            ax.plot(x, y, marker="o", markersize=1.5, color="black",
                    linestyle=None)
            return
        # Plot a point for current tree node
        x = [tree.data["age"]]
        y = [tree.data["mass"]]
        ax.plot(x, y, marker="o", markersize=1.5, color="black",
                linestyle=None)
        # Plot a line from current tree node to its main node
        x = [tree.data["age"], tree.main.data["age"]]
        y = [tree.data["mass"], tree.main.data["mass"]]
        ax.plot(x, y, color="blue")
        if tree.sub:
            # Plot a line between main and sub nodes
            x = [tree.main.data["age"], tree.sub.data["age"]]
            y = [tree.main.data["mass"], tree.sub.data["mass"]]
            ax.plot(x, y, color="green", linewidth=1, alpha=0.8)
        # Recursively plot the descendant nodes
        _plot(tree.main, ax)
        _plot(tree.sub, ax)

    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    _plot(mtree, ax=ax)
    ax.set_xlabel("Cosmic time [Gyr]")
    ax.set_ylabel("Mass [Msun]")
    ax.set_xlim((0, mtree.data["age"]))
    ax.set_ylim((0, mtree.data["mass"]))
    fig.tight_layout()
    canvas.print_figure(outfile)
