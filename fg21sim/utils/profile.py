# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Profiling utilities to help analyze the code.
"""

import sys
import resource
import gc
from numbers import Number
from collections import Set, Mapping, deque, defaultdict


def getsize(obj):
    """
    Recursively iterate to sum size of object & members.

    Returns
    -------
    size : int
        The size of the object in units of "Bytes".

    Credit
    ------
    * How do I determine the size of an object in Python?
      https://stackoverflow.com/a/30316760/4856091
    """
    zero_depth_bases = (str, bytes, Number, range, bytearray)

    def inner(obj, _seen_ids=set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0

        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            # bypass remaining control flow and return
            pass
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, "items"):
            size += sum(inner(k) + inner(v) for k, v in obj.items())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, "__dict__"):
            size += inner(vars(obj))
        if hasattr(obj, "__slots__"):
            # can have ``__slots__`` with ``__dict__``
            size += sum(inner(getattr(obj, s))
                        for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj)


def get_objcounts():
    """
    Get the current objects and counts as an dictionary.

    Credit
    ------
    * Working around memory leaks
      https://stackoverflow.com/a/1641280/4856091
    """
    objcounts = defaultdict(int)
    for obj in gc.get_objects():
        objcounts[type(obj)] += 1
    return objcounts


def diff_objcounts(objc, objc_ref):
    """
    Compare the ``objc1`` to ``objc_ref`` and return the differences
    in a list of ``(type, counts)``.

    Credit
    ------
    * Working around memory leaks
      https://stackoverflow.com/a/1641280/4856091
    """
    diff = [(k, objc[k]-objc_ref[k])
            for k in objc
            if (objc[k]-objc_ref[k])]
    return diff


def mem_usage(MiB=True):
    """
    Get the current memory usage.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # [KiB]
    unit = "KiB"
    if MiB:
        usage /= 1024
        unit = "MiB"
    return (usage, unit)
