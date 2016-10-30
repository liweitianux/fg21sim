# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Web user interface (UI) of "fg21sim" based upon Tornado_.

.. _Tornado: http://www.tornadoweb.org/
"""

import os

import tornado.web

from .websocket import EchoWSHandler


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


_settings = {
    # The static files will be served from the default "/static/" URI
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
    "template_path": os.path.join(os.path.dirname(__file__), "templates"),
}


def make_application(**kwargs):
    settings = _settings
    settings.update(kwargs)
    appplication = tornado.web.Application(
        handlers=[
            (r"/", IndexHandler),
            (r"/ws", EchoWSHandler),
        ], **settings)
    return appplication
