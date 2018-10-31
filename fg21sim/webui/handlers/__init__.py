# Copyright (c) 2016,2018 Weitian LI <wt@liwt.net>
# MIT license

from .index import IndexHandler  # noqa: F401
from .login import LoginHandler  # noqa: F401
from .configs import ConfigsAJAXHandler  # noqa: F401
from .console import ConsoleAJAXHandler  # noqa: F401
from .products import (  # noqa: F401
        ProductsAJAXHandler,
        ProductsDownloadHandler,
)
from .websocket import WSHandler  # noqa: F401
