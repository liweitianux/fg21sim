# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Web user interface (UI) of "fg21sim" based upon Tornado_ web server and
using the WebSocket_ protocol.

.. _Tornado: http://www.tornadoweb.org/

.. _WebSocket: https://en.wikipedia.org/wiki/WebSocket ,
   http://caniuse.com/#feat=websockets
"""

import os

import tornado.web

from .websocket import FG21simWSHandler
from ..configs import ConfigManager


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class Application(tornado.web.Application):
    configmanager = ConfigManager()

    def __init__(self, **kwargs):
        handlers = [
            (r"/",   IndexHandler),
            (r"/ws", FG21simWSHandler),
        ]
        settings = {
            # The static files will be served from the default "/static/" URI.
            # Recommend to use `{{ static_url(filepath) }}` in the templates.
            "static_path": os.path.join(os.path.dirname(__file__),
                                        "static"),
            "template_path": os.path.join(os.path.dirname(__file__),
                                          "templates"),
        }
        settings.update(kwargs)
        super().__init__(handlers, **settings)
