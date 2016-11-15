# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Handle the configurations operations with the client.
"""

import os
import logging

from ...errors import ConfigError


logger = logging.getLogger(__name__)


class ConfigsHandler:
    """
    Handle the "configs" type of messages from the client.
    """
    def __init__(self, configs):
        self.configs = configs

    def handle_message(self, msg):
        """
        Handle the message of type "configs", which request to get or
        set some configurations by the client.

        TODO:
        * improve the description ...
        * split these handling functions into a separate class in a module

        Parameters
        ----------
        msg : dict
            A dictionary parsed from the incoming JSON message, which
            generally has the following syntax:
            ``{"type": "configs", "action": <action>, "data": <data>}``
            where the ``<action>`` is ``set`` or ``get``, and the ``<data>``
            is a list of config keys or a dict of config key-value pairs.

        Returns
        -------
        response : dict
            A dictionary parsed from the incoming JSON message, which
            generally has the following syntax:
            ``{"type": "configs", "action": <action>,
               "data": <data>, "errors": <errors>}``
            where the ``<action>`` is the same as input, the ``<data>`` is
            a list of config keys or a dict of config key-value pairs, and
            ``<errors>`` contains the error message for the invalid config
            values.
        """
        try:
            msg_type = msg["type"]
            msg_action = msg["action"]
            response = {"type": msg_type, "action": msg_action}
            logger.info("WebSocket: handle message: " +
                        "type: {0}, action: {1}".format(msg_type, msg_action))
            if msg_action == "get":
                # Get the values of the specified options
                try:
                    data, errors = self._get_configs(keys=msg["keys"])
                    response["success"] = True
                    response["data"] = data
                    response["errors"] = errors
                except KeyError:
                    response["success"] = False
                    response["error"] = "'keys' is missing"
            elif msg_action == "set":
                # Set the values of the specified options
                try:
                    errors = self._set_configs(data=msg["data"])
                    response["success"] = True
                    response["data"] = {}  # be more consistent
                    response["errors"] = errors
                except KeyError:
                    response["success"] = False
                    response["error"] = "'data' is missing"
            elif msg_action == "reset":
                # Reset the configurations to the defaults
                self._reset_configs()
                response["success"] = True
            elif msg_action == "load":
                # Load the supplied user configuration file
                try:
                    success, error = self._load_configs(msg["userconfig"])
                    response["success"] = success
                    if not success:
                        response["error"] = error
                except KeyError:
                    response["success"] = False
                    response["error"] = "'userconfig' is missing"
            elif msg_action == "save":
                # Save current configurations to file
                try:
                    success, error = self._save_configs(msg["outfile"],
                                                        msg["clobber"])
                    response["success"] = success
                    if not success:
                        response["error"] = error
                except KeyError:
                    response["success"] = False
                    response["error"] = "'outfile' or 'clobber' is missing"
            else:
                logger.warning("WebSocket: " +
                               "unknown action: {0}".format(msg_action))
                response["success"] = False
                response["error"] = "unknown action: {0}".format(msg_action)
        except KeyError:
            # Received message has wrong syntax/format
            response = {"success": False,
                        "type": msg_type,
                        "error": "no action specified"}
        #
        logger.debug("WebSocket: response: {0}".format(response))
        return response

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

        NOTE
        ----
        Do not forget the ``userconfig`` option.
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
        """Set the values of the config options specified by the given keys
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
        errors : dict
            When error occurs (e.g., invalid key, invalid values), then the
            specific errors with details are stored in this dictionary.
        """
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
        # NOTE:
        # Check the whole configurations after all provided options are
        # updated, and merge the validation errors.
        __, cherr = self.configs.check_all(raise_exception=False)
        errors.update(cherr)
        return errors

    def _reset_configs(self):
        """Reset the configurations to the defaults."""
        self.configs.reset()

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
