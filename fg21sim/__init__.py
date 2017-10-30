"""
Realistic Foreground Simulation for 21 cm Reionization Signal Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016-2017 Weitian LI
:license: MIT
"""

__pkgname__ = "fg21sim"
__version__ = "0.6.3"
__author__ = "Weitian LI"
__author_email__ = "weitian@aaronly.me"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2016-2017 Weitian LI"
__url__ = "https://github.com/liweitianux/fg21sim"
__description__ = ("Realistic Foreground Simulation for "
                   "21 cm Reionization Signal Detection")


import logging


# Set a default logging handler to avoid the "No handler found" warning
logging.getLogger(__name__).addHandler(logging.NullHandler())
