# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
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
import operator
from functools import reduce
from collections import MutableMapping
import pkg_resources
import copy
import shutil

from configobj import ConfigObj, ConfigObjError, flatten_errors
from validate import Validator

from .checkers import check_configs
from ..errors import ConfigError


logger = logging.getLogger(__name__)


def _get_configspec():
    """Found and read all the configuration specifications"""
    files = sorted(pkg_resources.resource_listdir(__name__, ""))
    specfiles = [fn for fn in files if fn.endswith(".conf.spec")]
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


def _flatten_dict(d, sep="/", parent_key=""):
    """
    Recursively flatten a nested dictionary with keys compressed.

    The dictionary is recursively flattened into a one-level dictionary,
    i.e., all the leaves are raised to the top level.
    For each leaf, the value is simply preserved, while its keys list is
    compressed into a single key by concatenating with a separator.

    Parameters
    ----------
    d : dict
        Input nested dictionary.
    sep : str, optional
        The separator used to concatenate the keys for each leaf item.
    parent_key : str, optional
        The parent key string that will be prepended to the flatten key
        string when flattening the dictionary.
        NOTE:
        This parameter is required for the recursion, so user can simply
        ignore this..

    Returns
    -------
    flatdict : dict
        The flattened dictionary.

    Examples
    --------
    FIXME: fix the style
    - input nested dictionary:
      {'a': 1,
       'c': {'a': 2,
             'b': {'x': 5,
                   'y': 10}},
       'd': [1, 2, 3]}
    - output flatten dictionary:
      {'a': 1,
       'c/a': 2,
       'c/b/x': 5,
       'c/b/y': 10,
       'd': [1, 2, 3]}

    References
    ----------
    - Stackoverflow: Flatten nested Python dictionaries, compressing keys
      http://stackoverflow.com/a/6027615
    """
    items = []
    for k, v in d.items():
        new_key = (parent_key + sep + k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, sep=sep, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ConfigManager:
    """
    Manage the default configurations with specifications, as well as
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
        """
        Load the bundled default configurations and specifications.
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
        # NOTE: use ``copy.deepcopy``; see ``self.reset()`` for more details
        self._config = copy.deepcopy(self._config_default)
        if userconfig:
            self.read_userconfig(userconfig)

    def merge(self, config):
        """
        Simply merge the given configurations without validation.

        Parameters
        ----------
        config : `~ConfigObj`, dict, str, or list[str]
            Supplied configurations to be merged.
        """
        if not isinstance(config, ConfigObj):
            try:
                config = ConfigObj(config, interpolation=False)
            except ConfigObjError as e:
                logger.exception(e)
                raise ConfigError(e)
        self._config.merge(config)

    def read_config(self, config):
        """
        Read, validate and merge the input config.

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
        self.merge(newconfig)
        logger.info("Loaded additional config")

    def read_userconfig(self, userconfig):
        """
        Read user configuration file, validate, and merge into the
        default configurations.

        Parameters
        ----------
        userconfig : str
            Filename/path to the user configuration file.
            Generally, an absolute path should be provided.
            The prefix ``~`` (tilde) is also allowed and will be expanded.

        NOTE
        ----
        If a user configuration file is already loaded, then the
        configurations are *reset* before loading the supplied user
        configuration file.
        """
        userconfig = os.path.expanduser(userconfig)
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
        # NOTE:
        # * ``_config_default.copy()`` only returns a *shallow* copy.
        # * ``ConfigObj(_config_default)`` will lost all comments
        self._config = copy.deepcopy(self._config_default)
        self.userconfig = None
        logger.warning("Reset the configurations to the copy of defaults!")

    def _validate(self, config):
        """
        Validate the config against the specification using a default
        validator.  The validated config values are returned if success,
        otherwise, the ``ConfigError`` raised with details.
        """
        validator = Validator()
        try:
            # NOTE:
            # Use the "copy" mode, which will copy both the default values
            # and all the comments.
            results = config.validate(validator, preserve_errors=True,
                                      copy=True)
        except ConfigObjError as e:
            raise ConfigError(e)
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

    def check_all(self, raise_exception=True):
        """
        Further check the whole configurations through a set of custom
        checker functions, which may check one config option against its
        context if necessary to determine whether it has a valid value.

        Parameters
        ----------
        raise_exception : bool, optional
            Whether raise a ``ConfigError`` exception if there is any invalid
            config options?

        Returns
        -------
        result : bool
            ``True`` if the configurations pass all checker functions.
        errors : dict
            An dictionary containing the details about the invalid config
            options, with the keys identifying the config options and values
            indicating the error message.
            If above ``result=True``, then this is an empty dictionary ``{}``.

        Raises
        ------
        ConfigError
            If any config option failed to pass any of the checkers, a
            ``ConfigError`` with details is raised.
        """
        result, errors = check_configs(self, raise_exception=raise_exception)
        return (result, errors)

    def get(self, key, fallback=None, from_default=False):
        """Get config value by key."""
        if from_default:
            config = self._config_default
        else:
            config = self._config
        return config.get(key, fallback)

    def getn(self, key, from_default=False):
        """
        Get the value of a config option specified by the input key from
        from the configurations which is a nested dictionary.

        Parameters
        ----------
        key : str, or list[str]
            List of keys or a string of keys separated by a the ``/``
            character to specify the item in the ``self._config``, which
            is a nested dictionary.
            e.g., ``["section1", "key2"]``, ``"section1/key2"``
        from_default : bool, optional
            If True, get the config option value from the *default*
            configurations, other than the configurations merged with user
            configurations (default).

        Raises
        ------
        KeyError :
            The input key specifies a non-exist config option.

        References
        ----------
        - Stackoverflow: Checking a Dictionary using a dot notation string
          https://stackoverflow.com/q/12414821/4856091
          https://stackoverflow.com/a/12414913/4856091
        """
        if isinstance(key, str):
            key = key.split("/")
        #
        if from_default:
            config = self._config_default
        else:
            config = self._config
        #
        try:
            return reduce(operator.getitem, key, config)
        except (KeyError, TypeError):
            raise KeyError("%s: invalid key" % "/".join(key))

    def setn(self, key, value):
        """
        Set the value of config option specified by a list of keys or a
        "/"-separated keys string.

        The supplied key-value config pair is first used to create a
        temporary ``ConfigObj`` instance, which is then validated against
        the configuration specifications.
        If validated to be *valid*, the input key-value pair is then *merged*
        into the configurations, otherwise, a ``ConfigError`` raised.

        NOTE/XXX
        --------
        Given a ``ConfigObj`` instance with an option that does NOT exist in
        the specifications, it will simply *pass* the validation against the
        specifications.
        There seems no way to prevent the ``Validator`` from accepting the
        config options that does NOT exist in the specification.
        Therefore, try to get the option value specified by the input key
        first, if no ``KeyError`` raised, then it is a valid key.

        Parameters
        ----------
        key : str, or list[str]
            List of keys or a string of keys separated by a the ``/``
            character to specify the item in the ``self._config``, which
            is a nested dictionary.
            e.g., ``["section1", "key2"]``, ``"section1/key2"``
        value : str, bool, int, float, list
            The value can be any acceptable type to ``ConfigObj``.

        Raises
        ------
        KeyError :
            The input key specifies a non-exist config option.
        ConfigError :
            The value fails to pass the validation against specifications.
        """
        # NOTE:
        # May raise ``KeyError`` if the key does not exists
        val_old = self.getn(key)
        if val_old == value:
            # No need to set this option value
            return
        # Create a nested dictionary from the input key-value pair
        # Credit:
        # * Stackoverflow: Convert a list into a nested dictionary
        #   https://stackoverflow.com/a/6689604/4856091
        if isinstance(key, str):
            key = key.split("/")
        d = reduce(lambda x, y: {y: x}, reversed(key), value)
        # Create the temporary ``ConfigObj`` instance and validate it
        config_new = ConfigObj(d, interpolation=False,
                               configspec=self._configspec)
        # NOTE:
        # May raise ``ConfigError`` if fails to pass the validation
        config_new = self._validate(config_new)
        # NOTE:
        # The validated ``config_new`` is populated with all other options
        # from the specifications.
        val_new = reduce(operator.getitem, key, config_new)
        d2 = reduce(lambda x, y: {y: x}, reversed(key), val_new)
        self.merge(d2)
        logger.info("Set config: {key}: {val_new} <- {val_old}".format(
            key="/".join(key), val_new=val_new, val_old=val_old))

    def get_path(self, key):
        """
        Return the absolute path of the file/directory specified by the
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
        - The beginning ``~`` (tilde) is expanded to user's home directory.
        - The relative path (with respect to the user configuration file)
          is converted to absolute path if ``self.userconfig`` is valid.
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
            # Got a relative path, try to convert to the absolute path
            if self.userconfig is not None:
                # User configuration loaded
                path = os.path.join(os.path.dirname(self.userconfig), path)
            else:
                logger.warning("Cannot convert to absolute path: %s" % path)
        return os.path.normpath(path)

    @property
    def foregrounds(self):
        """
        Get all available and enabled foreground components.

        Returns
        -------
        enabled : list[str]
            Enabled foreground components to be simulated
        available : list[str]
            All available foreground components
        """
        fg = self.get("foregrounds")
        avaliable = list(fg.keys())
        enabled = [key for key, value in fg.items() if value]
        return (enabled, avaliable)

    @property
    def frequencies(self):
        """
        Get or calculate if ``frequency/type = custom`` the frequencies
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
        """
        Get and prepare the logging configurations for
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
        filemode = "a" if conf["appendmode"] else "w"
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

    def dump(self, from_default=False, flatten=False):
        """
        Dump the configurations as plain Python dictionary.

        Parameters
        ----------
        from_default : bool, optional
            If ``True``, dump the default configurations (as specified by the
            bundled specifications); otherwise, dump the configurations with
            user-supplied options merged (default).
        flatten : bool, optional
            If ``True``, flatten the configurations with is a nested
            dictionary into an one-level flat dictionary, make it much easier
            for client's manipulations.

        NOTE
        ----
        * The original option orders are missing.
        * The ``self.userconfig`` is also dumped.
        """
        if from_default:
            data = self._config_default.dict()
        else:
            data = self._config.dict()
        # Also dump the "userconfig" value
        data["userconfig"] = self.userconfig
        #
        if flatten:
            data = _flatten_dict(data)
        #
        return data

    def save(self, outfile=None, clobber=False, backup=True):
        """
        Save the configurations to file.

        Parameters
        ----------
        outfile : str, optional
            The path to the output configuration file.
            If not provided, then use ``self.userconfig``, however, set
            ``clobber=True`` may be required.
            NOTE:
            This must be an *absolute path*.
            Prefix ``~`` (tilde) is allowed and will be expanded.
        clobber : bool, optional
            Overwrite the output file if already exists.
        backup : bool, optional
            Backup the output file with suffix ``.old`` if already exists.

        Raises
        ------
        ValueError :
            The given ``outfile`` is not an *absolute path*, or the
            ``self.userconfig`` is invalid while the ``outfile`` not given.
        OSError :
            If the target filename already exists.
        """
        if outfile is None:
            if self.userconfig is None:
                raise ValueError("no outfile and self.userconfig is None")
            else:
                outfile = self.userconfig
                logger.warning("outfile not provided, " +
                               "use self.userconfig: {0}".format(outfile))
        outfile = os.path.expanduser(outfile)
        if not os.path.isabs(outfile):
            raise ValueError("not an absolute path: {0}".format(outfile))
        if os.path.exists(outfile):
            if clobber:
                # Make a backup with suffix ``.old``
                backfile = outfile + ".old"
                shutil.copyfile(outfile, backfile)
                logger.info("Backed up old configuration file as: " + backfile)
            else:
                raise OSError("outfile already exists: {0}".format(outfile))
        # Write out the configurations
        # NOTE: need open the output file in *binary* mode
        with open(outfile, "wb") as f:
            self._config.indent_type = "  "
            self._config.write(f)
