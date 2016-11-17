# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Custom checkers to further validate the configurations.

NOTE
----
These functions further check the configurations as a whole, which means
one config option may be checked against its context.
Therefore, they are very different to the checker functions used in the
``validate.Validator``.
"""

import os

from ..errors import ConfigError


def _check_missing(configs, keys):
    """Check whether the required config is provided by the user."""
    results = {}
    if isinstance(keys, str):
        keys = [keys, ]
    for key in keys:
        if not configs.getn(key):
            results[key] = "Value required but missing"
    return results


def _check_existence(configs, keys):
    """Check whether the file/directory corresponding to the config exists."""
    if isinstance(keys, str):
        keys = [keys, ]
    results = {}
    for key in keys:
        res = _check_missing(configs, key)
        if res == {}:
            # Both "key" and "dir_key" are valid
            path = configs.get_path(key)
            if not os.path.exists(path):
                res[key] = 'File/directory not exist: "%s"' % path
        results.update(res)
    return results


def _is_power2(n):
    """Check a number whether a power of 2"""
    # Credit: https://stackoverflow.com/a/29489026/4856091
    return (n and not (n & (n-1)))


def check_common(configs):
    """Check the "[common]" section of the configurations."""
    results = {}
    # "common/nside" must be a power of 2
    key = "common/nside"
    res = _check_missing(configs, key)
    if res == {}:
        if not _is_power2(configs.getn(key)):
            results[key] = "not a power of 2"
    else:
        results.update(res)
    # "common/lmax" must be greater than "common/lmin"
    key = "common/lmax"
    res = _check_missing(configs, [key, "common/lmin"])
    if res == {}:
        if configs.getn(key) <= configs.getn("common/lmin"):
            results[key] = "not greater than 'common/lmin'"
    else:
        results.update(res)
    # Check enabled components
    key = "common/components"
    if len(configs.getn(key)) == 0:
        results[key] = "no components enabled/selected"
    return results


def check_frequency(configs):
    """Check the "[frequency]" section of the configurations."""
    results = {}
    if configs.getn("frequency/type") == "custom":
        results.update(_check_missing(configs, "frequency/frequencies"))
    elif configs.getn("frequency/type") == "calc":
        results.update(
            _check_missing(configs, ["frequency/start",
                                     "frequency/stop",
                                     "frequency/step"])
        )
    return results


def check_output(configs):
    """Check the "[output]" section of the configurations."""
    results = {}
    if configs.getn("output/combine"):
        results.update(_check_missing(configs, "output/output_dir"))
    return results


def check_galactic_synchrotron(configs):
    """Check the "[galactic][synchrotron]" section of the configurations."""
    comp = "galactic/synchrotron"
    comp_enabled = configs.getn("common/components")
    results = {}
    if comp in comp_enabled:
        # Only validate the configs if this component is enabled
        results.update(
            _check_missing(configs, [comp+"/template_freq",
                                     comp+"/template_unit"])
        )
        results.update(
            _check_existence(configs, [comp+"/template", comp+"/indexmap"])
        )
        if configs.getn(comp+"/save"):
            results.update(_check_missing(configs, comp+"/output_dir"))
    return results


def check_galactic_freefree(configs):
    """Check the "[galactic][freefree]" section of the configurations."""
    comp = "galactic/freefree"
    comp_enabled = configs.getn("common/components")
    results = {}
    if comp in comp_enabled:
        # Only validate the configs if this component is enabled
        results.update(
            _check_missing(configs, [comp+"/halphamap_unit",
                                     comp+"/dustmap_unit"])
        )
        results.update(
            _check_existence(configs, [comp+"/halphamap", comp+"/dustmap"])
        )
        if configs.getn(comp+"/save"):
            results.update(_check_missing(configs, comp+"/output_dir"))
    return results


def check_galactic_snr(configs):
    """Check the "[galactic][snr]" section of the configurations."""
    comp = "galactic/snr"
    comp_enabled = configs.getn("common/components")
    results = {}
    if comp in comp_enabled:
        # Only validate the configs if this component is enabled
        results.update(
            _check_existence(configs, comp+"/catalog")
        )
        if configs.getn(comp+"/save"):
            results.update(_check_missing(configs, comp+"/output_dir"))
    return results


def check_extragalactic_clusters(configs):
    """
    Check the "[extragalactic][clusters]" section of the configurations.
    """
    comp = "extragalactic/clusters"
    comp_enabled = configs.getn("common/components")
    results = {}
    if comp in comp_enabled:
        # Only validate the configs if this component is enabled
        results.update(
            _check_existence(configs, comp+"/catalog")
        )
        if configs.getn(comp+"/save"):
            results.update(_check_missing(configs, comp+"/output_dir"))
    return results


# Available checkers to validate the configurations
_CHECKERS = [
    check_common,
    check_frequency,
    check_output,
    check_galactic_synchrotron,
    check_galactic_freefree,
    check_galactic_snr,
    check_extragalactic_clusters,
]


def check_configs(configs, raise_exception=True, checkers=_CHECKERS):
    """
    Check/validate the whole configurations through all the supplied
    checker functions.

    These checker functions may check one config option against its context
    if necessary to determine whether it has a valid value.

    Parameters
    ----------
    configs : `ConfigManager` instance
        An ``ConfigManager`` instance contains both default and user
        configurations.
    raise_exception : bool, optional
        Whether raise a ``ConfigError`` exception if there is any invalid
        config options?
    checkers : list of functions, optional
        List of checker functions through which the configurations
        will be checked.

    Returns
    -------
    validity : bool
        ``True`` if the configurations pass all checker functions.
    errors : dict
        An dictionary containing the details about the invalid config options,
        with the keys identifying the config options and values indicating
        the error message.
        If above ``validity=True``, then this is an empty dictionary ``{}``.

    Raises
    ------
    ConfigError :
        With ``raise_exception=True``, if any configuration option failed
        to pass all checkers, the ``ConfigError`` with details is raised.
    """
    errors = {}
    for checker in checkers:
        errors.update(checker(configs))
    #
    if errors == {}:
        validity = True
    else:
        validity = False
        if raise_exception:
            msg = "\n".join(['Config "{key}": {val}'.format(key=key, val=val)
                             for key, val in errors.items()])
            raise ConfigError(msg)
    return (validity, errors)
