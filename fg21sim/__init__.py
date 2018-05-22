__pkgname__ = "FG21sim"
__version__ = "0.7.1"
__date__ = "2018-01-16"
__author__ = "Weitian LI"
__author_email__ = "weitian@aaronly.me"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2016-2018 Weitian LI"
__url__ = "https://github.com/liweitianux/fg21sim"
__description__ = ("Foreground Simulation for "
                   "21 cm Reionization Signal Detection")


import logging


# Set a default logging handler to avoid the "No handler found" warning
logging.getLogger(__name__).addHandler(logging.NullHandler())
