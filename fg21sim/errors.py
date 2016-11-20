# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Custom errors/exceptions.
"""


class ConfigError(Exception):
    """Could not parse user configurations"""
    pass


class ManifestError(Exception):
    """Errors when build and/or manipulate the products manifest"""
    pass
