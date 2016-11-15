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
from tornado.options import options

from .console import ConsoleHandler
from .configs import ConfigsHandler
from ..utils import get_host_ip, ip_in_network


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
            logger.warning("WebSocket: %s: any hosts are allowed" % self.name)
        elif ip_in_network(ip, network):
            allow = True
            logger.info("WebSocket: %s: " % self.name +
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
        self.configs = self.application.configmanager
        self.console_handler = ConsoleHandler(websocket=self)
        self.configs_handler = ConfigsHandler(configs=self.configs)
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
                response = self.configs_handler.handle_message(msg)
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

    def _handle_results(self, msg):
        # Got a message of supported types
        msg_type = msg["type"]
        logger.info("WebSocket: {0}: ".format(self.name) +
                    "handle message of type: {0}".format(msg_type))
        response = {"success": True, "type": msg_type}
        return response
