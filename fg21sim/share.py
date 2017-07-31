# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Globally shared instances/objects shared throughout ``fg21sim``.

NOTE: ``global`` and ``globals`` are both preserved by Python :-(
"""

from .configs import ConfigManager
from .utils import Cosmology


# The globally shared `~ConfigManager` instance/object, that holds the
# default configurations as well as user-provided configurations.
# It may be imported by other modules to obtain the current effective
# configuration values, therefore greatly simplify the necessary parameters
# to be passed.
#
# NOTE: The entry script (e.g., `bin/fg21sim`) should load the user
#       configurations into this global object by e.g.,:
#       ``CONFIGS.read_userconfig(<user_config_file>)``
CONFIGS = ConfigManager()

# The globally shared `~Cosmology` instance/object may be used by other
# modules to calculate various cosmological quantities, and removes the
# need to pass the necessary/user cosmology parameters (e.g., H0, OmegaM0).
#
# NOTE: Once the above shared ``CONFIGS`` setup or loaded with new
#       configurations, this ``COSMO`` object needs also been updated:
#       ``COSMO.setup(**CONFIGS.cosmology)``
COSMO = Cosmology(**CONFIGS.cosmology)
