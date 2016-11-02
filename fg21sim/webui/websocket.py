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

from urllib.parse import urlparse
import logging

import tornado.websocket


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
    """
    name = "fg21sim"
    from_localhost = None

    def open(self):
        """Invoked when a new WebSocket is opened by the client."""
        logger.info("WebSocket: %s: opened" % self.name)

    def on_message(self, message):
        """Handle incoming messages."""
        logger.info("WebSocket: %s: received: %s" % (self.name, message))
        msg_back = message[::-1]
        logger.info("WebSocket: %s: sent: %s" % (self.name, msg_back))
        self.write_message(msg_back)

    def on_close(self):
        """Invoked when a new WebSocket is closed by the client."""
        if hasattr(self, "close_code"):
            code = self.close_code
        if hasattr(self, "close_reason"):
            reason = self.close_reason
        logger.info("WebSocket: {0}: closed by client: {1}, {2}".format(
            self.name, code, reason))

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
        logger.info("WebSocket: {0}: access origin: {1}".format(
            self.name, origin))
        # NOTE: `urlparse`: `scheme://netloc/path;parameters?query#fragment`
        netloc = urlparse(origin).netloc
        # NOTE: `netloc` example: `user:pass@example.com:8080`
        host = netloc.split("@")[-1].split(":")[0]
        if host in ["localhost", "127.0.0.1"]:
            self.from_localhost = True
            logger.info("WebSocket: %s: access from localhost" % self.name)
            return True
        # XXX/TODO: check whether from the local LAN ??
        return False
