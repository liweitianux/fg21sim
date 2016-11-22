# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Handle the configurations operations with the client.
"""

import os
import logging

import tornado.web
from tornado.escape import json_decode, json_encode

from .base import BaseRequestHandler
from ...errors import ConfigError


logger = logging.getLogger(__name__)


class ConfigsAJAXHandler(BaseRequestHandler):
    """
    Handle the AJAX requests from the client to manipulate the configurations.
    """
    def initialize(self):
        """Hook for subclass initialization.  Called for each request."""
        self.configs = self.application.configmanager

    def get(self):
        """
        Handle the READ-ONLY configuration manipulations.

        Supported actions:
        - get: Get the specified/all configuration values
        - validate: Validate the configurations and response the errors
        - exists: Whether the file already exists

        NOTE
        ----
        READ-WRITE configuration manipulations should be handled by
        the ``self.post()`` method.
        """
        action = self.get_argument("action", "get")
        data = {}
        errors = {}
        if action == "get":
            keys = json_decode(self.get_argument("keys", "null"))
            data, errors = self._get_configs(keys=keys)
            success = True
        elif action == "validate":
            __, errors = self.configs.check_all(raise_exception=False)
            success = True
        elif action == "exists":
            filepath = json_decode(self.get_argument("filepath", "null"))
            exists, error = self._exists_file(filepath)
            if exists is None:
                success = False
                reason = error
            else:
                success = True
                data["exists"] = exists
        else:
            # ERROR: bad action
            success = False
            reason = "Bad request action: {0}".format(action)
        #
        if success:
            response = {"action": action,
                        "data": data,
                        "errors": errors}
            logger.debug("Response: {0}".format(response))
            self.set_header("Content-Type", "application/json; charset=UTF-8")
            self.write(json_encode(response))
        else:
            logger.warning("Request failed: {0}".format(reason))
            self.send_error(400, reason=reason)

    @tornado.web.authenticated
    def post(self):
        """
        Handle the READ-WRITE configuration manipulations.

        Supported actions:
        - set: Set the specified configuration(s) to the posted value(s)
        - reset: Reset the configurations to its backup defaults
        - load: Load the supplied user configuration file
        - save: Save current configurations to file

        NOTE
        ----
        READ-ONLY configuration manipulations should be handled by
        the ``self.get()`` method.
        """
        request = json_decode(self.request.body)
        logger.debug("Received request: {0}".format(request))
        action = request.get("action")
        data = {}
        errors = {}
        if action == "set":
            # Set the values of the specified options
            try:
                data, errors = self._set_configs(request["data"])
                success = True
            except KeyError:
                success = False
                reason = "'data' is missing"
        elif action == "reset":
            # Reset the configurations to the defaults
            success = self._reset_configs()
        elif action == "load":
            # Load the supplied user configuration file
            try:
                success, reason = self._load_configs(request["userconfig"])
            except KeyError:
                success = False
                reason = "'userconfig' is missing"
        elif action == "save":
            # Save current configurations to file
            try:
                success, reason = self._save_configs(request["outfile"],
                                                     request["clobber"])
            except KeyError:
                success = False
                reason = "'outfile' or 'clobber' is missing"
        else:
            # ERROR: bad action
            success = False
            reason = "Bad request action: {0}".format(action)
        #
        if success:
            response = {"action": action,
                        "data": data,
                        "errors": errors}
            logger.debug("Response: {0}".format(response))
            self.set_header("Content-Type", "application/json; charset=UTF-8")
            self.write(json_encode(response))
        else:
            logger.warning("Request failed: {0}".format(reason))
            self.send_error(400, reason=reason)

    def _get_configs(self, keys=None):
        """Get the values of the config options specified by the given keys.

        Parameters
        ----------
        keys : list[str], optional
            A list of keys specifying the config options whose values will
            be obtained.
            If ``keys=None``, then all the configurations values are dumped.

        Returns
        -------
        data : dict
            A dictionary with keys the same as the input keys, and values
            the corresponding config option values.
        errors : dict
            When error occurs (e.g., invalid key), then the specific errors
            with details are stored in this dictionary.
        """
        if keys is None:
            # Dump all the configurations
            data = self.configs.dump(flatten=True)
            data["userconfig"] = self.configs.userconfig
            errors = {}
        else:
            data = {}
            errors = {}
            for key in keys:
                if key == "userconfig":
                    data["userconfig"] = self.configs.userconfig
                else:
                    try:
                        data[key] = self.configs.getn(key)
                    except KeyError as e:
                        errors[key] = str(e)
        #
        return (data, errors)

    def _set_configs(self, data):
        """
        Set the values of the config options specified by the given keys
        to the corresponding supplied data.

        NOTE
        ----
        The ``userconfig`` needs special handle.
        The ``workdir`` and ``configfile`` options should be ignored.

        Parameters
        ----------
        data : dict
            A dictionary of key-value pairs, with keys specifying the config
            options whose value will be changed, and values the new values
            to which config options will be set.
            NOTE:
            If want to set the ``userconfig`` option, an *absolute path*
            must be provided.

        Returns
        -------
        data_orig : dict
            When the supplied value failed to pass the specification
            validation, then its original value was returned to the client
            to reset its value.
        errors : dict
            When error occurs (e.g., invalid key, invalid values), then the
            specific errors with details are stored in this dictionary.
        """
        data_orig = {}
        errors = {}
        for key, value in data.items():
            if key in ["workdir", "configfile"]:
                # Ignore "workdir" and "configfile"
                continue
            elif key == "userconfig":
                if os.path.isabs(os.path.expanduser(value)):
                    self.configs.userconfig = value
                else:
                    errors[key] = "Not an absolute path"
            else:
                try:
                    self.configs.setn(key, value)
                except KeyError as e:
                    errors[key] = str(e)
                except ConfigError as e:
                    data_orig[key] = self.configs.getn(key)
                    errors[key] = str(e)
        #
        return (data_orig, errors)

    def _reset_configs(self):
        """Reset the configurations to the defaults."""
        self.configs.reset()
        return True

    def _load_configs(self, userconfig):
        """Load configurations from the provided user configuration file.

        Parameters
        ----------
        userconfig: str
            The filepath to the user configuration file, which must be
            an *absolute path*.

        Returns
        -------
        success : bool
            ``True`` if the operation succeeded, otherwise, ``False``.
        error : str
            If failed, this ``error`` saves the details, otherwise, ``None``.
        """
        success = False
        error = None
        if os.path.isabs(os.path.expanduser(userconfig)):
            try:
                self.configs.read_userconfig(userconfig)
                success = True
            except ConfigError as e:
                error = str(e)
        else:
            error = "Not an absolute path"
        return (success, error)

    def _save_configs(self, outfile, clobber=False):
        """Save current configurations to file.

        Parameters
        ----------
        outfile: str
            The filepath to the output configuration file, which must be
            an *absolute path*.
        clobber : bool, optional
            Whether overwrite the output file if already exists?

        Returns
        -------
        success : bool
            ``True`` if the operation succeeded, otherwise, ``False``.
        error : str
            If failed, this ``error`` saves the details, otherwise, ``None``.
        """
        success = False
        error = None
        try:
            self.configs.save(outfile, clobber=clobber)
            success = True
        except (ValueError, OSError) as e:
            error = str(e)
        return (success, error)

    @staticmethod
    def _exists_file(filepath):
        """Check whether the given filepath already exists?

        Parameters
        ----------
        filepath: str
            The input filepath to check its existence, must be
            an *absolute path*.

        Returns
        -------
        exists : bool
            ``True`` if the filepath already exists, ``False`` if not exists,
            and ``None`` if error occurs.
        error : str
            The error information, and ``None`` if no errors.
        """
        exists = None
        error = None
        try:
            filepath = os.path.expanduser(filepath)
            if os.path.isabs(filepath):
                exists = os.path.exists(filepath)
            else:
                error = "Not an absolute path: {0}".format(filepath)
        except AttributeError:
            error = "Invalid input filepath: {0}".format(filepath)
        return (exists, error)
