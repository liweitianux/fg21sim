"""
Realistic Foregrounds Simulation for EoR 21cm Signal Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 Weitian LI
:license: MIT
"""

__pkgname__ = "fg21sim"
__version__ = "0.3.0"
__author__ = "Weitian LI"
__author_email__ = "liweitianux@live.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2016 Weitian LI"
__url__ = "https://github.com/liweitianux/fg21sim"
__description__ = ("Realistic Foregrounds Simulation for "
                   "EoR 21cm Signal Detection")


import logging


# Set default logging handle to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
