# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Index page handler
"""

from .base import BaseRequestHandler


class IndexHandler(BaseRequestHandler):
    """
    Index page handler of the Web UI.
    """
    def get(self):
        self.render("index.html")
