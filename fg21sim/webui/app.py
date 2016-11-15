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
from tornado.web import url

from .handlers import IndexHandler, LoginHandler, FG21simWSHandler
from .utils import gen_cookie_secret
from ..configs import ConfigManager


class Application(tornado.web.Application):
    configmanager = ConfigManager()

    def __init__(self, **kwargs):
        handlers = [
            url(r"/", IndexHandler, name="index"),
            url(r"/login", LoginHandler, name="login"),
            url(r"/ws", FG21simWSHandler),
        ]
        settings = {
            # The static files will be served from the default "/static/" URI.
            # Recommend to use `{{ static_url(filepath) }}` in the templates.
            "static_path": os.path.join(os.path.dirname(__file__),
                                        "static"),
            "template_path": os.path.join(os.path.dirname(__file__),
                                          "templates"),
            # URL to be redirected to if the user is not logged in
            "login_url": r"/login",
            # Secret key used to sign the cookies
            "cookie_secret": gen_cookie_secret(),
            # Enable "cross-site request forgery" (XSRF)
            "xsrf_cookies": True,
        }
        settings.update(kwargs)
        super().__init__(handlers, **settings)
