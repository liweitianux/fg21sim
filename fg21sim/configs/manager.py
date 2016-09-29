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
from glob import glob
import logging

from configobj import ConfigObj, ConfigObjError, flatten_errors
from validate import Validator


class ConfigError(Exception):
    """Could not parse user configurations"""
    pass


CONFIGS_PATH = os.path.dirname(__file__)


class ConfigManager:
    """Manager the configurations"""
    def __init__(self, configs=None):
        """
        :param configs: (optional) list of user configs to load
        """
        configs_spec = sorted(glob(os.path.join(CONFIGS_PATH, "*.conf.spec")))
        spec = "\n".join([open(f).read() for f in configs_spec]).split("\n")
        self._configspec = ConfigObj(spec, interpolation=False,
                                     list_values=False, _inspec=True)
        configs_default = ConfigObj(interpolation=False,
                                    configspec=self._configspec)
        self._config = self.validate(configs_default)
        if configs:
            for config in configs:
                self.read_config(config)

    def read_config(self, config):
        newconfig = ConfigObj(config, interpolation=False,
                              configspec=self._configspec)
        newconfig = self.validate(newconfig)
        self._config.merge(newconfig)

    def validate(self, config):
        validator = Validator()
        try:
            results = config.validate(validator, preserve_errors=True)
        except ConfigObjError as e:
            raise ConfigError(e.message)
        if not results:
            error_msg = ''
            for (section_list, key, res) in flatten_errors(config, results):
                if key is not None:
                    if res is False:
                        msg = 'key "%s" in section "%s" is missing.'
                        msg = msg % (key, ', '.join(section_list))
                    else:
                        msg = 'key "%s" in section "%s" failed validation: %s'
                        msg = msg % (key, ', '.join(section_list), res)
                else:
                    msg = 'section "%s" is missing' % '.'.join(section_list)
                error_msg += msg + '\n'
            raise ConfigError(error_msg)
        return config

    def get(self, key, fallback=None):
        if key in self._config:
            value = self._config[key]
        else:
            value = fallback
        return value

    def set(self, key, value):
        self._config[key] = value

    @property
    def logging(self):
        """Get and prepare the logging configurations for
        `logging.basicConfig()`
        """
        from logging import FileHandler, StreamHandler
        conf = self.get("logging")
        # logging handlers
        handlers = []
        stream = conf["stream"]
        if stream:
            handlers.append(StreamHandler(getattr(sys, stream)))
        logfile = conf["filename"]
        if logfile:
            handlers.append(FileHandler(logfile, mode=conf["filemode"]))
        #
        logconf = {
            "level": getattr(logging, conf["level"]),
            "format": conf["format"],
            "datefmt": conf["datefmt"],
            "handlers": handlers,
        }
        return logconf
