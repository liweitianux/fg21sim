# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
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


def check_foregrounds(configs):
    """Check the "[foregrounds]" section of the configurations."""
    results = {}
    # Check enabled foreground components
    fg = configs.foregrounds
    if len(fg[0]) == 0:
        results["foregrounds"] = "no foreground components enabled"
    return results


def check_sky(configs):
    """Check the "[sky]" section of the configurations."""
    results = {}
    skytype = configs.getn("sky/type")
    if skytype == "patch":
        sec = "sky/patch"
    elif skytype == "healpix":
        sec = "sky/healpix"
        # "nside" must be a power of 2
        key = sec + "/nside"
        res = _check_missing(configs, key)
        if res == {}:
            if not _is_power2(configs.getn(key)):
                results[key] = "not a power of 2"
        else:
            results.update(res)
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


def check_galactic_synchrotron(configs):
    """Check the "[galactic][synchrotron]" section of the configurations."""
    comp = "galactic/synchrotron"
    comp_enabled = configs.foregrounds[0]
    if comp not in comp_enabled:
        return {}

    results = {}
    # Only validate the configs if this component is enabled
    results.update(
        _check_missing(configs, comp+"/template_freq")
    )
    results.update(
        _check_existence(configs, [comp+"/template", comp+"/indexmap"])
    )
    if configs.getn(comp+"/add_smallscales"):
        # "lmax" must be greater than "lmin"
        key = comp + "/lmax"
        res = _check_missing(configs, [key, comp+"/lmin"])
        if res == {}:
            if configs.getn(key) <= configs.getn(comp+"/lmin"):
                results[key] = "not greater than 'lmin'"
        else:
            results.update(res)
    results.update(_check_missing(configs, comp+"/output_dir"))
    return results


def check_galactic_freefree(configs):
    """Check the "[galactic][freefree]" section of the configurations."""
    comp = "galactic/freefree"
    comp_enabled = configs.foregrounds[0]
    results = {}
    if comp in comp_enabled:
        # Only validate the configs if this component is enabled
        results.update(
            _check_existence(configs, [comp+"/halphamap", comp+"/dustmap"])
        )
        results.update(_check_missing(configs, comp+"/output_dir"))
    return results


def check_galactic_snr(configs):
    """Check the "[galactic][snr]" section of the configurations."""
    comp = "galactic/snr"
    comp_enabled = configs.foregrounds[0]
    results = {}
    if comp in comp_enabled:
        # Only validate the configs if this component is enabled
        results.update(_check_existence(configs, comp+"/catalog"))
        results.update(_check_missing(configs, comp+"/output_dir"))
    return results


def check_extragalactic_clusters(configs):
    """
    Check the "[extragalactic][clusters]" section of the configurations.
    The related sections ("[extragalactic][psformalism]",
    "[extragalactic][halos]") are also checked.
    """
    comp = "extragalactic/clusters"
    comp_enabled = configs.foregrounds[0]
    results = {}
    if comp in comp_enabled:
        # output dndlnm data file required
        key = "extragalactic/psformalism/dndlnm_outfile"
        results.update(_check_missing(configs, key))
        # catalog required when enabled to use it
        if configs.get(comp+"/use_output_catalog"):
            results.update(_check_existence(configs, comp+"/catalog_outfile"))
        results.update(_check_missing(configs, comp+"/output_dir"))
    return results


# Available checkers to validate the configurations
_CHECKERS = [
    check_foregrounds,
    check_sky,
    check_frequency,
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
    configs : `~ConfigManager`
        An ``ConfigManager`` instance contains both default and user
        configurations.
    raise_exception : bool, optional
        Whether raise a ``ConfigError`` exception if there is any invalid
        config options?
    checkers : list[function], optional
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
