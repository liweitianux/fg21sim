# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Index page handler
"""

from .base import BaseRequestHandler


class IndexHandler(BaseRequestHandler):
    """
    Index page handler of the Web UI.

    Attributes
    ----------
    from_localhost : bool
        ``True`` if the request is from the localhost, otherwise ``False``.
    """
    def initialize(self):
        """Hook for subclass initialization.  Called for each request."""
        if self.request.remote_ip == "127.0.0.1":
            self.from_localhost = True
        else:
            self.from_localhost = False

    def get(self):
        self.render("index.html", from_localhost=self.from_localhost)
