# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Custom validations for the configurations.
"""

from ..errors import ConfigError


def _check_missing(configs, keys):
    """Check whether the mandatory config is provided by the user."""
    if isinstance(keys, str):
        keys = [keys, ]
    for key in keys:
        if not configs.getn(key):
            raise ConfigError('config "%s" missing' % key)
    return True


def check_common(configs):
    """Check the "[common]" section of the configurations."""
    _check_missing(configs, "common/data_dir")
    return True


def check_frequency(configs):
    """Check the "[frequency]" section of the configurations."""
    if configs.getn("frequency/type") == "custom":
        _check_missing(configs, "frequency/frequencies")
    elif configs.getn("frequency/type") == "calc":
        _check_missing(configs, ["frequency/start",
                                 "frequency/stop",
                                 "frequency/step"])
    return True


def check_output(configs):
    """Check the "[output]" section of the configurations."""
    if configs.getn("output/combine"):
        _check_missing(configs, "output/output_dir")
    return True


def check_galactic_synchrotron(configs):
    """Check the "[galactic][synchrotron]" section of the configurations."""
    _check_missing(configs, ["galactic/synchrotron/template",
                             "galactic/synchrotron/template_freq",
                             "galactic/synchrotron/template_unit",
                             "galactic/synchrotron/indexmap"])
    if configs.getn("galactic/synchrotron/save"):
        _check_missing(configs, "galactic/synchrotron/output_dir")
    return True


# Available checkers to validate the configurations
_CHECKERS = [
    check_common,
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
    for checker in checkers:
        checker(configs)
    return True
