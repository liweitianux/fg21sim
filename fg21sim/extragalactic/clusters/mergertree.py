# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license

"""
Merger tree that represents the merging history of a cluster using
the binary tree data structure.
"""

import os
import pickle
import logging


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
    merged : `~MergerTree`
        Reverse link to the merged cluster.  Therefore, it is able to refer
        to the ``sub`` cluster from the ``main`` cluster, and vice versa.
    """
    def __init__(self, data, main=None, sub=None, merged=None):
        self.data = data
        self.main = main
        self.sub = sub
        self.merged = merged


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


def plot_mtree(mtree):
    """
    Plot the cluster merger tree.
    """
    raise NotImplementedError
