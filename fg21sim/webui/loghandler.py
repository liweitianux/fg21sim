# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Custom logging handlers

WebSocketLogHandler :
    Send logging messages to the WebSocket as JSON-encoded string.
"""


import logging
import json


class WebSocketLogHandler(logging.Handler):
    """
    Send logging messages to the WebSocket as JSON-encoded string.

    Parameters
    ----------
    websocket : `~tornado.websocket.WebSocketHandler`
        An `~tornado.websocket.WebSocketHandler` instance, which has
        the ``write_message()`` method that will be used to send the
        logging messages.
    msg_type : str, optional
        Set the type of the sent back message, for easier processing
        by the client.

    NOTE
    ----
    The message sent through the WebSocket is a JSON-encoded string
    from a dictionary, e.g.,
    ``{"type": self.msg_type,
       "action": "log",
       "levelname": record.levelname,
       "levelno": record.levelno,
       "name": record.name,
       "asctime": record.asctime,
       "message": <formatted-message>}``
    """
    def __init__(self, websocket, msg_type=None):
        super().__init__()
        self.websocket = websocket
        self.msg_type = msg_type

    def emit(self, record):
        try:
            message = self.format(record)
            msg = json.dumps({
                "type": self.msg_type,
                "action": "log",
                "levelname": record.levelname,
                "levelno": record.levelno,
                "name": record.name,
                "asctime": record.asctime,
                "message": message,
            })
            self.websocket.write_message(msg)
        except Exception:
            self.handleError(record)
