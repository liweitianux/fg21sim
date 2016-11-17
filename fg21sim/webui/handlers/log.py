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
    Push the logging messages to the client(s) through the WebSocket(s)
    as JSON-encoded strings.

    Parameters
    ----------
    websockets : set of `~tornado.websocket.WebSocketHandler`
        Set of opened websockets, through which the logging messages will
        be pushed.
    msg_type : str, optional
        Set the type of the pushed logging messages, for easier handling
        by the client.

    NOTE
    ----
    The pushed logging message is a JSON-encoded string from a dictionary:
    ``{"type": self.msg_type,
       "subtype": "log",
       "action": "push",
       "levelname": record.levelname,
       "levelno": record.levelno,
       "name": record.name,
       "asctime": record.asctime,
       "message": <formatted-message>}``
    """
    def __init__(self, websockets, msg_type=None):
        super().__init__()
        self.websockets = websockets
        self.msg_type = msg_type

    def emit(self, record):
        try:
            message = self.format(record)
            msg = json.dumps({
                "type": self.msg_type,
                "subtype": "log",
                "action": "push",
                "levelname": record.levelname,
                "levelno": record.levelno,
                "name": record.name,
                "asctime": record.asctime,
                "message": message,
            })
            for ws in self.websockets:
                ws.write_message(msg)
        except Exception:
            self.handleError(record)
