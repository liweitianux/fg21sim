# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Login handler
"""

import tornado.web

from .base import BaseRequestHandler


class IndexHandler(BaseRequestHandler):
    """
    Index page handler of the Web UI.
    """
    @tornado.web.authenticated
    def get(self):
        self.render("index.html")
