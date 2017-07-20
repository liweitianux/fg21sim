"""
Realistic Foregrounds Simulation for EoR 21cm Signal Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016-2017 Weitian LI
:license: MIT
"""

__pkgname__ = "fg21sim"
__version__ = "0.4.2"
__author__ = "Weitian LI"
__author_email__ = "weitian@aaronly.me"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2016-2017 Weitian LI"
__url__ = "https://code.aaronly.me/fg21sim.git"
__description__ = ("Realistic Foregrounds Simulation for "
                   "EoR 21cm Signal Detection")


import logging


# Set default logging handle to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
