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

import logging

import tornado.websocket
from tornado.escape import json_encode
from tornado.options import options

from ..utils import get_host_ip, ip_in_network


logger = logging.getLogger(__name__)


class WSHandler(tornado.websocket.WebSocketHandler):
    """
    Push messages (e.g., logging messages, configurations) to the client.

    NOTE
    ----
    WebSocket is a bi-directional and real-time communication protocol, which
    is great for active messages pushing.
    However, WebSocket is a rather low-level protocol.  It receives and sends
    messages independently, so it does not provide any support of
    request-response operations, RPC (remote-procedure call), etc.
    Therefore, it is hard/problematic to implement some interactions similar
    to the traditional AJAX techniques.

    There exists some high-level sub-protocols built upon the WebSocket, e.g.,
    WAMP [1]_, which provides better features and are easier to use, allowing
    to fully replace the AJAX etc. techniques.
    However, the Tornado (v4.3) currently does not support them, and the
    corresponding client JavaScript tool is also required.

    XXX/WARNING
    -----------
    ``WebSocket.on_message()``: may NOT be a coroutine at the moment (v4.3).
    See [2]_ and [3]_ .

    Attributes
    ----------
    from_localhost : bool
        Set to ``True`` if the access is from the localhost,
        otherwise ``False``.

    References
    ----------
    .. _[1] WAMP: Web Application Messaging Protocl, http://wamp-proto.org/
    .. _[2] https://stackoverflow.com/a/35543856/4856091
    .. _[3] https://stackoverflow.com/a/33724486/4856091
    """
    from_localhost = None

    def check_origin(self, origin):
        """
        Check the origin of the WebSocket connection to determine whether
        the access is allowed.

        Attributes
        ----------
        from_localhost : bool
            Set to ``True`` if the access is from the "localhost" (i.e.,
            127.0.0.1), otherwise ``False``.
        """
        self.from_localhost = False
        logger.info("WebSocket: origin: {0}".format(origin))
        ip = get_host_ip(url=origin)
        network = options.hosts_allowed
        if ip == "127.0.0.1":
            self.from_localhost = True
            allow = True
            logger.info("WebSocket: origin is 'localhost'")
        elif network.upper() == "ANY":
            # Any hosts are allowed
            allow = True
            logger.warning("WebSocket: ANY hosts are allowed")
        elif ip_in_network(ip, network):
            allow = True
            logger.info("WebSocket: client from allowed network: %s" % network)
        else:
            allow = False
            logger.error("WebSocket: " +
                         "client is NOT in the allowed network: %s" % network)
        return allow

    def open(self):
        """Invoked when a new WebSocket is opened by the client."""
        # Add to the set of current connected clients
        self.application.websockets.add(self)
        logger.info("Added new opened WebSocket client: {0}".format(self))
        self.configs = self.application.configmanager
        # Push current configurations to the client
        self._push_configs()

    def on_close(self):
        """Invoked when a new WebSocket is closed by the client."""
        # Remove from the set of current connected clients
        self.application.websockets.remove(self)
        logger.warning("Removed closed WebSocket client: {0}".format(self))

    def broadcast(self, message):
        """Broadcast/push the given message to all connected clients."""
        for ws in self.application.websockets:
            ws.write_message(message)

    def _push_configs(self):
        """
        Get the current configurations as well as the validation status,
        then push to the client to updates the configurations form.
        """
        data = self.configs.dump(flatten=True)
        data["userconfig"] = self.configs.userconfig
        __, errors = self.configs.check_all(raise_exception=False)
        msg = {"success": True,
               "type": "configs",
               "action": "push",
               "data": data,
               "errors": errors}
        message = json_encode(msg)
        logger.debug("Message of current configurations: {0}".format(message))
        self.write_message(message)
        logger.info("WebSocket: Pushed current configurations data " +
                    "with validation errors to the client")
