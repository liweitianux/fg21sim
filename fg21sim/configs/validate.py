# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Custom validations for the configurations.

NOTE
----
These checker functions check the configurations as a whole, and may check
a config item against its context,
Therefore, they are very different to the checker function of `Validator`.
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
    results = {}
    results.update(
        _check_missing(configs, ["galactic/synchrotron/template_freq",
                                 "galactic/synchrotron/template_unit"])
    )
    results.update(
        _check_existence(configs, ["galactic/synchrotron/template",
                                   "galactic/synchrotron/indexmap"])
    )
    if configs.getn("galactic/synchrotron/save"):
        results.update(_check_missing(configs,
                                      "galactic/synchrotron/output_dir"))
    return results


# Available checkers to validate the configurations
_CHECKERS = [
    check_frequency,
    check_output,
    check_galactic_synchrotron,
]


def validate_configs(configs, checkers=_CHECKERS):
    """Validate the configurations through the supplied checkers.

    These checker usually validate on the global scale, and validate
    some specific configs against their contexts.

    Parameters
    ----------
    configs : `ConfigManager` object
        An `ConfigManager` object contains both default and user
        configurations.
    checkers : list of functions
        List of checker functions through which the configurations
        will be validated.

    Returns
    -------
    bool
        True if the configurations pass all checker functions, otherwise,
        the `ConfigError` will be raised with corresponding message.

    Raises
    ------
    ConfigError
        If any configuration failed the check, a `ConfigError` with
        details will be raised.
    """
    results = {}
    for checker in checkers:
        results.update(checker(configs))
    #
    if results == {}:
        return True
    else:
        err_msg = "\n".join(['Config "{key}": {msg}'.format(key=key, msg=msg)
                             for key, msg in results.items()])
        raise ConfigError(err_msg)
