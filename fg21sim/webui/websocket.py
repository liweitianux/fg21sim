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

import json
import logging

import tornado.websocket

from ..configs import ConfigManager
from .utils import get_host_ip


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
    configs = ConfigManager()

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
        logger.info("WebSocket: {0}: origin: {1}".format(self.name, origin))
        ip = get_host_ip(url=origin)
        if ip == "127.0.0.1":
            self.from_localhost = True
            logger.info("WebSocket: %s: origin is localhost" % self.name)
            return True
        else:
            self.from_localhost = False
            # FIXME/TODO: check whether from local LAN (or in same subnet)??
            logger.error("WebSocket: %s: " % self.name +
                         "ONLY allow access from localhost at the moment :(")
            return False

    def open(self):
        """Invoked when a new WebSocket is opened by the client."""
        logger.info("WebSocket: %s: opened" % self.name)

    def on_close(self):
        """Invoked when a new WebSocket is closed by the client."""
        code, reason = None, None
        if hasattr(self, "close_code"):
            code = self.close_code
        if hasattr(self, "close_reason"):
            reason = self.close_reason
        logger.info("WebSocket: {0}: closed by client: {1}, {2}".format(
            self.name, code, reason))

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
        There is a ``status`` item which indicates the status of the
        requested operation, and a ``data`` item containing the response
        data.
        """
        logger.debug("WebSocket: %s: received: %s" % (self.name, message))
        try:
            msg = json.loads(message)
            msg_type = msg["type"]
        except json.JSONDecodeError:
            logger.warning("WebSocket: {0}: ".format(self.name) +
                           "message is not a valid JSON string")
            response = {"status": False, "type": None}
        except (KeyError, TypeError):
            logger.warning("WebSocket: %s: skip invalid message" % self.name)
            response = {"status": False, "type": None}
        else:
            # Check the message type and dispatch task
            if msg_type == "configs":
                # Request or set the configurations
                response = self._handle_configs(msg)
            elif msg_type == "console":
                # Control the simulation tasks, or request logging messages
                response = self._handle_console(msg)
            elif msg_type == "results":
                # Request the simulation results
                response = self._handle_results(msg)
            else:
                # Message of unknown type
                logger.warning("WebSocket: {0}: ".format(self.name) +
                               "unknown message type: {0}".format(msg_type))
                response = {"status": False, "type": msg_type}
        #
        msg_response = json.dumps(response)
        self.write_message(msg_response)

    def _handle_configs(self, msg):
        """Handle the message of type "configs", which request to get or
        set some configurations by the client.

        TODO: improve the description ...

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
            msg_data = msg["data"]
            response = {"type": msg_type, "action": msg_action}
            logger.info("WebSocket: {0}: handle message: ".format(self.name) +
                        "type: {0}, action: {1}".format(msg_type, msg_action))
            if msg_action == "get":
                # Get the values of the specified options
                data, errors = self._get_configs(keys=msg_data)
                response["status"] = True if errors == {} else False
                response["data"] = data
                response["errors"] = errors
            elif msg_action == "set":
                # Set the values of the specified options
                errors = self._set_configs(data=msg_data)
                response["status"] = True if errors == {} else False
                response["data"] = {}
                response["errors"] = errors
            else:
                logger.warning("WebSocket: {0}: ".format(self.name) +
                               "unknown action: {0}".format(msg_action))
                response["status"] = False
                response["data"] = {}
                response["errors"] = {}
        except KeyError:
            # Received message has wrong syntax/format
            response = {"status": False, "type": msg_type, "action": None}
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
            data = self.configs.dump()
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

        Parameters
        ----------
        data : dict
            A dictionary of key-value pairs, with keys specifying the config
            options whose value will be changed, and values the new values
            to which config options will be set.

        Returns
        -------
        errors : dict
            When error occurs (e.g., invalid key, invalid values), then the
            specific errors with details are stored in this dictionary.
        """
        pass

    def _handle_console(self, msg):
        # Got a message of supported types
        msg_type = msg["type"]
        logger.info("WebSocket: {0}: ".format(self.name) +
                    "handle message of type: {0}".format(msg_type))
        response = {"status": True, "type": msg_type}
        return response

    def _handle_results(self, msg):
        # Got a message of supported types
        msg_type = msg["type"]
        logger.info("WebSocket: {0}: ".format(self.name) +
                    "handle message of type: {0}".format(msg_type))
        response = {"status": True, "type": msg_type}
        return response
