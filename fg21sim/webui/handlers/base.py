# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Base web request handler for other handlers
"""


import tornado.web
from tornado.options import options


class BaseRequestHandler(tornado.web.RequestHandler):
    """
    Base web request handler with user authentication support.
    """

    def get_current_user(self):
        """
        Override the ``get_current_user()`` method to implement user
        authentication.

        Determine the current user based on the value of a cookie.

        References
        ----------
        - Tornado: Authentication and security
          http://www.tornadoweb.org/en/stable/guide/security.html
        """
        if (options.password is None) or (options.password == ""):
            # Password not set, then all accesses are allowed
            return True
        else:
            return self.get_secure_cookie("user")
