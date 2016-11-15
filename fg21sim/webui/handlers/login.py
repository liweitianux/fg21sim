# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Login handler
"""

from tornado.options import options
from tornado.escape import xhtml_escape

from .base import BaseRequestHandler


class LoginHandler(BaseRequestHandler):
    """
    Login page handler of the Web UI.

    NOTE
    ----
    Only check the password to authenticate the access, therefore, the
    default username "FG21SIM" is used.
    """
    def get(self):
        if (options.password is None) or (options.password == ""):
            # Password is not set, just allow
            self.redirect(self.reverse_url("index"))
        elif self.current_user:
            # Already authenticated
            self.redirect(self.reverse_url("index"))
        else:
            self.render("login.html", error="")

    def post(self):
        password = xhtml_escape(self.get_argument("password"))
        if password == options.password:
            self.set_secure_cookie("user", "FG21SIM")
            self.redirect(self.reverse_url("index"))
        else:
            # Password incorrect
            self.render("login.html", error="Incorrect password!")
