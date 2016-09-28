"""
Realistic Foregrounds Simulation for EoR 21cm Signal Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 Weitian LI
:license: MIT
"""

__pkgname__ = "fg21sim"
__version__ = "0.0.1"
__author__ = "Weitian LI"
__author_email__ = "liweitianux@live.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2016 Weitian LI"
__url__ = "https://github.com/liweitianux/fg21sim"
__description__ = ("Realistic Foregrounds Simulation for "
                   "EoR 21cm Signal Detection")


# Set default logging handle to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
