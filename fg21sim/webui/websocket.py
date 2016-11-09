# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Communicate with the "fg21sim" simulation program through the Web UI using
the WebSocket_ protocol, which provides full-duplex communication channels
over a single TCP connection.

.. _WebSocket: https://en.wikipedia.org/wiki/WebSocket


References
----------
- Tornado WebSocket:
  http://www.tornadoweb.org/en/stable/websocket.html
- Can I Use: WebSocket:
  http://caniuse.com/#feat=websockets
"""

import os
import json
import logging

import tornado.websocket
from tornado.options import options

from .consolehandler import ConsoleHandler
from .utils import get_host_ip, ip_in_network
from ..configs import ConfigManager
from ..errors import ConfigError


logger = logging.getLogger(__name__)


class FG21simWSHandler(tornado.websocket.WebSocketHandler):
    """
    WebSocket for bi-directional communication between the Web UI and
    the server, which can deal with the configurations and execute the
    simulation task.

    Generally, WebSocket send and receive data as *string*.  Therefore,
    the more complex data are stringified as JSON string before sending,
    which will be parsed after receive.

    Each message (as a JSON object or Python dictionary) has a ``type``
    field which will be used to determine the following action to take.

    Attributes
    ----------
    name : str
        Name to distinguish this WebSocket handle.
    from_localhost : bool
        Set to ``True`` if the access is from the localhost,
        otherwise ``False``.
    configs : `~ConfigManager`
        A ``ConfigManager`` instance, for configuration manipulations when
        communicating with the Web UI.
    """
    name = "fg21sim"
    from_localhost = None

    def check_origin(self, origin):
        """Check the origin of the WebSocket access.

        Attributes
        ----------
        from_localhost : bool
            Set to ``True`` if the access is from the localhost,
            otherwise ``False``.

        NOTE
        ----
        Currently, only allow access from the ``localhost``
        (i.e., 127.0.0.1) and local LAN.
        """
        self.from_localhost = False
        logger.info("WebSocket: {0}: origin: {1}".format(self.name, origin))
        ip = get_host_ip(url=origin)
        network = options.hosts_allowed
        if ip == "127.0.0.1":
            self.from_localhost = True
            allow = True
            logger.info("WebSocket: %s: origin is localhost" % self.name)
        elif network.upper() == "ANY":
            # Any hosts are allowed
            allow = True
            logger.error("WebSocket: %s: any hosts are allowed" % self.name)
        elif ip_in_network(ip, network):
            allow = True
            logger.error("WebSocket: %s: " % self.name +
                         "client is in the allowed network: %s" % network)
        else:
            allow = False
            logger.error("WebSocket: %s: " % self.name +
                         "client is NOT in the allowed network: %s" % network)
        return allow

    def open(self):
        """Invoked when a new WebSocket is opened by the client."""
        # FIXME:
        # * better to move to the `Application` class ??
        # * or create a ``ConfigsHandler`` similar to the ``ConsoleHandler``
        self.configs = ConfigManager()
        self.console_handler = ConsoleHandler(websocket=self)
        #
        logger.info("WebSocket: {0}: opened".format(self.name))
        logger.info("Allowed hosts: {0}".format(options.hosts_allowed))

    def on_close(self):
        """Invoked when a new WebSocket is closed by the client."""
        code, reason = None, None
        if hasattr(self, "close_code"):
            code = self.close_code
        if hasattr(self, "close_reason"):
            reason = self.close_reason
        logger.info("WebSocket: {0}: closed by client: {1}, {2}".format(
            self.name, code, reason))

    # FIXME/XXX:
    # * How to be non-blocking ??
    # NOTE: WebSocket.on_message: may NOT be a coroutine at the moment (v4.3)
    # References:
    # [1] https://stackoverflow.com/a/35543856/4856091
    # [2] https://stackoverflow.com/a/33724486/4856091
    def on_message(self, message):
        """Handle incoming messages and dispatch task according to the
        message type.

        NOTE
        ----
        The received message (parsed to a Python dictionary) has a ``type``
        item which will be used to determine the following action to take.

        Currently supported message types are:
        ``configs``:
            Request or set the configurations
        ``console``:
            Control the simulation tasks, or request logging messages
        ``results``:
            Request the simulation results

        The sent message also has a ``type`` item of same value, which the
        client can be used to figure out the proper actions.
        There is a ``success`` item which indicates the status of the
        requested operation, and an ``error`` recording the error message
        if ``success=False``.
        """
        logger.debug("WebSocket: %s: received: %s" % (self.name, message))
        try:
            msg = json.loads(message)
            msg_type = msg["type"]
        except json.JSONDecodeError:
            logger.warning("WebSocket: {0}: ".format(self.name) +
                           "message is not a valid JSON string")
            response = {"success": False,
                        "type": None,
                        "error": "message is not a valid JSON string"}
        except (KeyError, TypeError):
            logger.warning("WebSocket: %s: skip invalid message" % self.name)
            response = {"success": False,
                        "type": None,
                        "error": "type is missing"}
        else:
            # Check the message type and dispatch task
            if msg_type == "configs":
                # Request or set the configurations
                response = self._handle_configs(msg)
            elif msg_type == "console":
                # Control the simulation tasks, or request logging messages
                # FIXME/XXX:
                # * How to make this asynchronously ??
                response = self.console_handler.handle_message(msg)
            elif msg_type == "results":
                # Request the simulation results
                response = self._handle_results(msg)
            else:
                # Message of unknown type
                logger.warning("WebSocket: {0}: ".format(self.name) +
                               "unknown message type: {0}".format(msg_type))
                response = {"success": False,
                            "type": msg_type,
                            "error": "unknown message type %s" % msg_type}
        #
        msg_response = json.dumps(response)
        self.write_message(msg_response)

    def _handle_configs(self, msg):
        """Handle the message of type "configs", which request to get or
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
            logger.info("WebSocket: {0}: handle message: ".format(self.name) +
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
                logger.warning("WebSocket: {0}: ".format(self.name) +
                               "unknown action: {0}".format(msg_action))
                response["success"] = False
                response["error"] = "unknown action: {0}".format(msg_action)
        except KeyError:
            # Received message has wrong syntax/format
            response = {"success": False,
                        "type": msg_type,
                        "error": "no action specified"}
        #
        logger.debug("WebSocket: {0}: ".format(self.name) +
                     "response: {0}".format(response))
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
        The ``userconfig`` and ``workdir`` options are ignored.

        Parameters
        ----------
        data : dict
            A dictionary of key-value pairs, with keys specifying the config
            options whose value will be changed, and values the new values
            to which config options will be set.
            NOTE:
            If want to set the ``userconfig`` option, an *absolute path*
            must be provided (i.e., client should take care of the ``workdir``
            value and generate a absolute path for ``userconfig``).

        Returns
        -------
        errors : dict
            When error occurs (e.g., invalid key, invalid values), then the
            specific errors with details are stored in this dictionary.
        """
        errors = {}
        for key, value in data.items():
            if key in ["userconfig", "workdir"]:
                # Ignore "userconfig" and "workdir"
                continue
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
        if os.path.isabs(userconfig):
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

    def _handle_results(self, msg):
        # Got a message of supported types
        msg_type = msg["type"]
        logger.info("WebSocket: {0}: ".format(self.name) +
                    "handle message of type: {0}".format(msg_type))
        response = {"success": True, "type": msg_type}
        return response
