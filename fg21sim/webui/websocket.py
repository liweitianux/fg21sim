# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Communicate with the "fg21sim" simulation program through the Web UI using
the WebSocket_ technique, which provides full-duplex communication channels
over a single TCP connection.

.. _WebSocket: https://en.wikipedia.org/wiki/WebSocket ,
   http://caniuse.com/#feat=websockets
"""

import tornado.websocket


class EchoWSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print("WebSocket opened")

    def on_message(self, message):
        print("Message received: %s" % message)
        msg_back = message[::-1]
        print("Message sent back: %s" % msg_back)
        self.write_message(msg_back)

    def on_close(self):
        print("WebSocket closed")
