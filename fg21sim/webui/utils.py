# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Utilities for the Web UI
------------------------

gen_cookie_secret :
    Generate the secret key for cookie signing from the local hostname.
"""


import socket
import base64


def gen_cookie_secret():
    """
    Generate the secret key for cookie signing from the local hostname.
    """
    hostname = socket.gethostname()
    secret = base64.b64encode(hostname.encode("utf-8")).decode("ascii")
    return secret
