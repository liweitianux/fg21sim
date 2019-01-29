# Copyright (c) 2017-2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Merger tree that represents the merging history of a cluster using
the binary tree data structure.
"""

import operator as op
import logging

from ...utils.io import pickle_dump, pickle_load

logger = logging.getLogger(__name__)


class MergerTree:
    """
    A binary tree that represents the cluster merging history.

    Description
    -----------
                   Merged (M0, z0, age0)
                   ~~~~~~~~~~~~~~~~~~~~~
                   |                   |
        Main (M1, z1, age1)    Sub (M2, z2, age2)
        ~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~

    * "Merged" is the merged cluster from "Main" and "Sub" at redshift
      z1 (=z2) or cosmic time age1 (=age2).
      M0 = M1 + M2, M1 > M2
    * If "Sub" is missing, then this is an accretion event (not a merger).

    Parameters
    ----------
    data : any (e.g., a dictionary)
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

    def itermain(self):
        """
        Iterate by tracing the main cluster.
        """
        maintree = self
        subtree = None
        while maintree:
            main = maintree.data
            if subtree is not None:
                sub = subtree.data
            else:
                sub = {}
            yield (main, sub)
            subtree = maintree.sub
            maintree = maintree.main

    @property
    def lmain(self):
        """
        Return the length (i.e., number of events) along the main cluster.
        """
        return len(list(self.itermain()))

    def imain(self, idx):
        """
        Trace the main cluster to locate the idx-th (0-based) event, and
        return its data (both the main and sub clusters).

        Parameters
        ----------
        idx : int
            The event index (0-based) along the main cluster to be retrieved.

        Returns
        -------
        (main, sub) : dict
            The data dictionaries of the main and sub clusters.  The ``sub``
            dictionary may be ``None`` if it is an accretion event.
        """
        for main, sub in self.itermain():
            if idx == 0:
                return (main, sub)
            idx -= 1
        raise IndexError("index out of range: %d" % idx)


def save_mtree(mtree, outfile, clobber=False):
    """
    Pickle the merger tree data and save to file.
    """
    pickle_dump(mtree, outfile=outfile, clobber=clobber)
    logger.info("Saved merger tree to file: {0}".format(outfile))


def read_mtree(infile):
    mtree = pickle_load(infile)
    logger.info("Loaded merger tree from file: {0}".format(infile))
    return mtree


def get_history(mtree, merger_only=False):
    """
    Extract all the formation history (merger and accretion events).

    Parameters
    ----------
    mtree : `~MergerTree`
        The simulated merger tree from which to extract the history.
        Default: ``self.mtree``
    merger_only : bool, optional
        If ``True``, only extract the merger events.

    Returns
    -------
    evlist : list[event]
        List of events with each element being a dictionary of the
        event properties.
    """
    evlist = []
    for main, sub in mtree.itermain():
        z, age, M_main = op.itemgetter("z", "age", "mass")(main)
        if sub:
            # merger
            M_sub = sub["mass"]
            R_mass = M_main / M_sub
        else:
            # accretion
            if merger_only:
                continue
            M_sub, R_mass = None, None

        evlist.append({
            "z": z,
            "age": age,
            "M_main": M_main,
            "M_sub": M_sub,
            "R_mass": R_mass,
        })

    return evlist


def show_mtree(mtree):
    """
    Trace the main cluster and show its formation history.
    """
    parent = None
    for i, (main, sub) in enumerate(mtree.itermain()):
        info = "%2d: " % i
        z = main["z"]
        age = main["age"]
        info += "<z=%.3f; t=%5.2f> " % (z, age)
        mass = main["mass"]
        if sub:
            # merger event
            Msub = sub["mass"]
            Rmass = mass / Msub
            info += "[%.3e @@@ %.3e] (Rm=%5.1f)" % (mass, Msub, Rmass)
        elif parent:
            # accretion event
            dM = parent["mass"] - mass
            info += " %.3e  +  %.3e            " % (mass, dM)
        else:
            # root cluster
            info += " %.3e" % mass
        if parent:
            info += " <dz=%.3f/dt=%.2f>" % (z-parent["z"], parent["age"]-age)
        parent = main
        print(info)


def plot_mtree(mtree, outfile, figsize=(12, 8)):
    """
    Plot the cluster merger tree.

    XXX: Need to speed up this function.

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

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    print("Plotting the merger tree, may take a while ...")
    _plot(mtree, ax=ax)
    ax.set_xlabel("Cosmic time [Gyr]")
    ax.set_ylabel("Mass [Msun]")
    ax.set_xlim((0, mtree.data["age"]))
    ax.set_ylim((0, mtree.data["mass"]))
    fig.tight_layout()
    canvas.print_figure(outfile)
    print("Saved plot to file: %s" % outfile)
