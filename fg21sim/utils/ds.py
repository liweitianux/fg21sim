# Copyright (c) 2017,2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Data structure/set utilities.
"""

import logging
from collections import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


def _flatten_list(l):
    """
    Flatten an arbitrarily nested list.

    Credit
    ------
    * Flatten (an irregular) list of lists
      https://stackoverflow.com/a/2158532
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten_list(el)
        else:
            yield el


def dictlist_to_dataframe(dictlist, keys=None):
    """
    Convert the data in format of list of dictionaries to be a Pandas
    DataFrame by flattening the dictionary keys into columns.

    NOTE
    ----
    If the item ``key`` of the dictionary has value of a list/vector,
    then it is split into multiple columns named as ``key[0], key[1], ...``.

    Parameters
    ----------
    dictlist : list[dict]
        The input data to be converted, is a list of dictionaries, with
        each member dictionary has the same format/structure.
        NOTE: The dictionary may have items with list/vector as the values,
              but other more complex items (e.g., nested dictionary) is not
              allowed and supported.
    keys : list[str], optional
        The list of dictionary items to be selected for conversion.
        Default: convert all dictionary items.

    Returns
    -------
    dataframe : `~pandas.DataFrame`
        The converted Pandas DataFrame with columns be the dictionary
        item keys.
    """
    d0 = dictlist[0]
    if keys is None:
        keys = list(d0.keys())
    logger.info("DataFrame conversion selected keys: {0}".format(keys))

    columns = []
    for k in keys:
        v = d0[k]
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            columns += ["%s[%d]" % (k, i) for i in range(len(v))]
        else:
            columns.append(k)
    logger.info("DataFrame number of columns: %d" % len(columns))
    logger.debug("DataFrame columns: {0}".format(columns))

    data = []
    for d in dictlist:
        dv = [d[k] for k in keys]
        dv2 = list(_flatten_list(dv))
        data.append(dv2)

    return pd.DataFrame(data, columns=columns)


def pad_dict_list(d, keys, length, fill=None):
    """
    Pad the lists specified by ``keys`` to the given ``length``.

    Parameters
    ----------
    d : dict
        The dictionary to be updated.
    keys : list[str]
        The keys of the lists in the dictionary to be padded.
    length : int
        The expected length.
    fill : optional
        The value filled to the padded list.
    """
    for k in keys:
        v = d[k]
        n = len(v)
        if n < length:
            d[k] = v + [fill] * (length - n)
