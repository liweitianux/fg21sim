# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Web user interface (UI) of "fg21sim" based upon Tornado_ web server.

.. _Tornado: http://www.tornadoweb.org/
"""

import os
from concurrent.futures import ThreadPoolExecutor

import tornado.web
from tornado.web import url
from tornado.options import define, options

from .handlers import (IndexHandler,
                       LoginHandler,
                       ConfigsAJAXHandler,
                       ConsoleAJAXHandler,
                       ProductsAJAXHandler,
                       ProductsDownloadHandler,
                       WSHandler)
from .utils import gen_cookie_secret
from ..configs import ConfigManager
from ..products import Products


# Each module defines its own options, which are added to the global namespace
define("max_workers", default=1, type=int,
       help="Maximum number of threads to execute the submitted tasks")


class Application(tornado.web.Application):
    """
    Application of the "fg21sim" Web UI.

    Attributes
    ----------
    configmanager : `~fg21sim.configs.ConfigManager`
        A ``ConfigManager`` instance, which saves the current configurations
        status.  The configuration operations (e.g., "set", "get", "load")
        are performed on this instance, which is also passed to the
        foregrounds simulation programs.
    websockets : set of `~tornado.websocket.WebSocketHandler`
        Current active WebSockets opened by clients.
        When a new WebSocket connection established, it is added to this
        list, which is also removed from this list when the connection lost.
    executor : `~concurrent.futures.ThreadPoolExecutor`
        An executor that uses a pool of threads to execute the submitted
        tasks asynchronously.
    task_status : dict
        Whether the task is running and/or finished?
        1. running=False, finished=False: not started
        2. running=False, finished=True:  finished
        3. running=True,  finished=False: running
        4. running=True,  finished=True:  ?? error ??
    products : `~fg21sim.products.Products`
        Manage and manipulate the simulation products
    """

    def __init__(self, **kwargs):
        self.configmanager = ConfigManager()
        self.websockets = set()
        self.executor = ThreadPoolExecutor(max_workers=options.max_workers)
        self.task_status = {"running": False, "finished": False}
        self.products = Products()
        # URL handlers
        handlers = [
            url(r"/", IndexHandler, name="index"),
            url(r"/login", LoginHandler, name="login"),
            url(r"/ajax/configs", ConfigsAJAXHandler),
            url(r"/ajax/console", ConsoleAJAXHandler),
            url(r"/ajax/products", ProductsAJAXHandler),
            url(r"/products/download/(.*)", ProductsDownloadHandler),
            url(r"/ws", WSHandler),
        ]
        if options.debug:
            from .handlers.base import BaseRequestHandler
            handlers.append(url(r"/debug", BaseRequestHandler, name="debug"))
        # Application settings
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
            # Enable "cross-site request forgery" (XSRF) protection
            "xsrf_cookies": True,
        }
        settings.update(kwargs)
        super().__init__(handlers, **settings)
