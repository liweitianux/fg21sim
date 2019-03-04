__pkgname__ = "FG21sim"
__version__ = "0.9.0"
__date__ = "2019-03-04"
__author__ = "Weitian LI"
__author_email__ = "wt@liwt.net"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2016-2019 Weitian LI"
__url__ = "https://github.com/liweitianux/fg21sim"
__description__ = ("Foreground Simulation for "
                   "21 cm Reionization Signal Detection")


import logging


# Set a default logging handler to avoid the "No handler found" warning
logging.getLogger(__name__).addHandler(logging.NullHandler())
