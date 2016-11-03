# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license
#
# References:
# [1] https://configobj.readthedocs.io/en/latest/configobj.html
# [2] https://github.com/pazz/alot/blob/master/alot/settings/manager.py

"""
Configuration manager.
"""

import os
import sys
import logging
from logging import FileHandler, StreamHandler
from functools import reduce
import pkg_resources

from configobj import ConfigObj, ConfigObjError, flatten_errors
from validate import Validator

from ..errors import ConfigError


logger = logging.getLogger(__name__)


def _get_configspec():
    """Found and read all the configuration specifications"""
    files = sorted(pkg_resources.resource_listdir(__name__, ""))
    # NOTE:
    # Explicit convert the filter results to a list, since the returned
    # iterator can ONLY be used ONCE.
    specfiles = list(filter(lambda fn: fn.endswith(".conf.spec"), files))
    if os.environ.get("DEBUG_FG21SIM"):
        print("DEBUG: Found config specifications: %s" % ", ".join(specfiles),
              file=sys.stderr)
    # NOTE:
    # `resource_string()` returns the resource in *binary/bytes* string
    configspec = "\n".join([
        pkg_resources.resource_string(__name__, fn).decode("utf-8")
        for fn in specfiles
    ]).split("\n")
    return configspec


class ConfigManager:
    """Manage the default configurations with specifications, as well as
    the user configurations.

    Both the default configurations and user configurations are validated
    against the bundled specifications.

    Parameters
    ----------
    userconfig: str, optional
        Filename/path to the user configuration file.
        If provided, the user configurations is also loaded, validated, and
        merged into the configurations data.
        The user configuration can also be later loaded by
        ``self.read_userconfig()``.

    Attributes
    ----------
    _config : `~configobj.ConfigObj`
        The current effective configurations.
    _configspec : `~configobj.ConfigObj`
        The configuration specifications bundled with this package.
    userconfig : str
        The filename and path to the user-provided configurations.
        NOTE:
        - This attribute has valid value only after loading the user
          configuration by ``self.read_userconfig()``;
        - This attribute is used to determine the absolute path of the
          configs specifying the input templates or data files, therefore
          allow the use of relative path for those config options.
    """
    # Path to the user provided configuration file, which indicates user
    # configurations merged if not ``None``.
    userconfig = None

    def __init__(self, userconfig=None):
        """Load the bundled default configurations and specifications.
        If the ``userconfig`` provided, the user configurations is also
        loaded, validated, and merged.
        """
        configspec = _get_configspec()
        self._configspec = ConfigObj(configspec, interpolation=False,
                                     list_values=False, _inspec=True)
        configs_default = ConfigObj(interpolation=False,
                                    configspec=self._configspec)
        # Keep a copy of the default configurations
        self._config_default = self._validate(configs_default)
        # NOTE: `_config_default.copy()` only returns a *shallow* copy.
        self._config = ConfigObj(self._config_default, interpolation=False)
        if userconfig:
            self.read_userconfig(userconfig)

    def read_config(self, config):
        """Read, validate and merge the input config.

        Parameters
        ----------
        config : str, or list[str]
            Input config to be validated and merged.
            This parameter can be the filename of the config file, or a list
            contains the lines of the configs.
        """
        try:
            newconfig = ConfigObj(config, interpolation=False,
                                  configspec=self._configspec)
        except ConfigObjError as e:
            raise ConfigError(e)
        newconfig = self._validate(newconfig)
        self._config.merge(newconfig)
        logger.info("Loaded additional config: {0}".format(config))

    def read_userconfig(self, userconfig):
        """Read user configuration file, validate, and merge into the
        default configurations.

        Parameters
        ----------
        userconfig : str
            Filename/path to the user configuration file.

        NOTE
        ----
        If a user configuration file is already loaded, then the
        configurations are *reset* before loading the supplied user
        configuration file.
        """
        try:
            config = open(userconfig).read().split("\n")
        except IOError:
            raise ConfigError('Cannot read config from "%s"' % userconfig)
        #
        if self.userconfig is not None:
            logger.warning('User configuration already loaded from "%s"' %
                           self.userconfig)
            self.reset()
        self.read_config(config)
        self.userconfig = os.path.abspath(userconfig)
        logger.info("Loaded user config: {0}".format(self.userconfig))

    def reset(self):
        """Reset the current configurations to the copy of defaults from
        the specifications.

        NOTE: Also reset ``self.userconfig`` to ``None``.
        """
        # NOTE: `_config_default.copy()` only returns a *shallow* copy.
        self._config = ConfigObj(self._config_default, interpolation=False)
        self.userconfig = None
        logger.warning("Reset the configurations to the copy of defaults!")

    def _validate(self, config):
        """Validate the config against the specification using a default
        validator.  The validated config values are returned if success,
        otherwise, the ``ConfigError`` raised with details.
        """
        validator = Validator()
        try:
            results = config.validate(validator, preserve_errors=True)
        except ConfigObjError as e:
            raise ConfigError(e.message)
        if results is not True:
            error_msg = ""
            for (section_list, key, res) in flatten_errors(config, results):
                if key is not None:
                    if res is False:
                        msg = 'key "%s" in section "%s" is missing.'
                        msg = msg % (key, ", ".join(section_list))
                    else:
                        msg = 'key "%s" in section "%s" failed validation: %s'
                        msg = msg % (key, ", ".join(section_list), res)
                else:
                    msg = 'section "%s" is missing' % ".".join(section_list)
                error_msg += msg + "\n"
            raise ConfigError(error_msg)
        return config

    def get(self, key, fallback=None, from_default=False):
        """Get config value by key."""
        if from_default:
            config = self._config_default
        else:
            config = self._config
        return config.get(key, fallback)

    def getn(self, key, sep="/", from_default=False):
        """Get the config value from the nested dictionary configs using
        a list of keys or a "sep"-separated keys strings.

        Parameters
        ----------
        key : str, or list[str]
            List of keys or a string separated by a specific character
            (e.g., "/") to specify the item in the ``self._config``, which
            is a nested dictionary.
            e.g., ``["section1", "key2"]``, ``"section1/key2"``
        sep : str (len=1), optional
            If the above "keys" is a string, then this parameter specify
            the character used to separate the multi-level keys.
            This parameter should be a string of length 1 (i.e., a character).
        from_default : bool, optional
            If True, get the config option value from the *default*
            configurations, other than the configurations merged with user
            configurations (default).

        References
        ----------
        - Stackoverflow: Checking a Dictionary using a dot notation string
          https://stackoverflow.com/q/12414821/4856091
        """
        if len(sep) != 1:
            raise ValueError("Invalid parameter 'sep': %s" % sep)
        if isinstance(key, str):
            key = key.split(sep)
        #
        if from_default:
            config = self._config_default
        else:
            config = self._config
        #
        return reduce(dict.get, key, config)

    def get_path(self, key):
        """Return the absolute path of the file/directory specified by the
        config keys.

        Parameters
        ----------
        key : str
            "/"-separated string specifying the config name of the
            file/directory

        Returns
        -------
        path : str
            The absolute path (if user configuration loaded) or relative
            path specified by the input key, or ``None`` if specified
            config is ``None``.

        Raises
        ------
        ValueError:
            If the value of the specified config is not string.

        NOTE
        ----
        - The "~" (tilde) inside path is expanded to the user home directory.
        - The relative path (with respect to the user configuration file)
          is converted to absolute path if `self.userconfig` presents.
        """
        value = self.getn(key)
        if value is None:
            logger.warning("Specified config '%s' is None or not exist" % key)
            return None
        if not isinstance(value, str):
            msg = "Specified config '%s' is non-string: %s" % (key, value)
            logger.error(msg)
            raise ValueError(msg)
        #
        path = os.path.expanduser(value)
        if not os.path.isabs(path):
            # Got relative path, try to convert to the absolute path
            if hasattr(self, "userconfig"):
                # User configuration loaded
                path = os.path.join(os.path.dirname(self.userconfig), path)
            else:
                logger.warning("Cannot convert to absolute path: %s" % path)
        return os.path.normpath(path)

    @property
    def frequencies(self):
        """Get or calculate if ``frequency/type = custom`` the frequencies
        where to perform the simulations.

        Returns
        -------
        frequencies : list[float]
            List of frequencies where the simulations are requested.
        """
        if self.getn("frequency/type") == "custom":
            # The value is validated to be a float list
            frequencies = self.getn("frequency/frequencies")
        else:
            # Calculate the frequency values
            start = self.getn("frequency/start")
            stop = self.getn("frequency/stop")
            step = self.getn("frequency/step")
            num = int((stop - start) / step + 1)
            frequencies = [start + step*i for i in range(num)]
        return frequencies

    @property
    def logging(self):
        """Get and prepare the logging configurations for
        ``logging.basicConfig()`` to initialize the logging module.

        NOTE
        ----
        ``basicConfig()`` will automatically create a ``Formatter`` with the
        giving ``format`` and ``datefmt`` for each handlers if necessary,
        and then adding the handlers to the "root" logger.
        """
        conf = self.get("logging")
        level = conf["level"]
        if os.environ.get("DEBUG_FG21SIM"):
            print("DEBUG: Force 'DEBUG' logging level", file=sys.stderr)
            level = "DEBUG"
        # logging handlers
        handlers = []
        stream = conf["stream"]
        if stream:
            handlers.append(StreamHandler(getattr(sys, stream)))
        logfile = conf["filename"]
        filemode = conf["filemode"]
        if logfile:
            handlers.append(FileHandler(logfile, mode=filemode))
        #
        logconf = {
            "level": getattr(logging, level),
            "format": conf["format"],
            "datefmt": conf["datefmt"],
            "filemode": filemode,
            "handlers": handlers,
        }
        return logconf

    def dump(self, from_default=False):
        """Dump the configurations as plain Python dictionary.

        Parameters
        ----------
        from_default : bool, optional
            If True, dump the default configurations (as specified by the
            bundled specifications); otherwise, dump the configurations with
            user-supplied options merged (default).

        NOTE
        ----
        The original option orders are missing.
        """
        if from_default:
            config = self._config_default
        else:
            config = self._config
        return config.dict()
