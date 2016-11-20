# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# This file contains the general configurations, which control the general
# behaviors, or will be used in other configuration sections.


# Common/general configurations for the simulation
[common]
# HEALPix Nside value, i.e., pixel resolution
# NOTE: also update "lmax" below.
nside = integer(min=1, default=1024)

# Range of multipole monents (l) of the angular power spectrum.
# The power spectrum will be cut off to a constant for multipole l < lmin.
# Generally, lmax = 3 * nside - 1
lmin = integer(min=0, default=10)
lmax = integer(min=1, default=3071)

# List of foreground components to be simulated:
# + galactic/synchrotron:
#       Diffuse Galactic synchrotron emission (unpolarized)
# + galactic/freefree:
#       Diffuse Galactic free-free emission
# + galactic/snr:
#       Galactic supernova remnants emission
components = force_list(default=list("galactic/synchrotron", "galactic/freefree", "galactic/snr", "extragalactic/clusters"))


# Frequencies specification of the simulation products
[frequency]
# Unit of the frequency value
unit = option("MHz", default="MHz")

# How to specify the frequencies
# + custom:
#       directly specify the frequency values using the "frequencies" config
# + calc:
#       calculate the frequency values by "start", "stop", and "step"
type = option("custom", "calc", default="custom")

# The frequency values to be simulated if above "type" is "custom".
frequencies = float_list(default=list())

# Parameters to calculate the frequencies
# start and stop frequency value (both inclusive)
start = float(default=None, min=0.0)
stop = float(default=None, min=0.0)
step = float(default=None, min=0.0)


# Configuration for output products
[output]
# Unit of the sky map pixel value
unit = option("K", default="K")

# Use single-precision float instead of double (also save spaces)
use_float = boolean(default=True)

# Filetype used to store the products (default: fits)
filetype = option("fits", default="fits")

# Filename pattern (without extension) for the output products, which will
# be finally formatted using `str.format()`.
filename_pattern = string(default="{prefix}_{frequency:05.1f}")

# Whether calculate the checksum for the output file (e.g., "CHECKSUM"
# keyword in FITS header)?
# NOTE:
# FITS checksum calculation may account for half the time to output the data.
checksum = boolean(default=False)

# Whether overwrite existing files
clobber = boolean(default=False)

# Whether combine all components and output
combine = boolean(default=True)
# Prefix for the combined files
combine_prefix = string(default="fg")
# Output directory to place the combined products
# NOTE: This config is mandatory and should be provided by the user
#       if above "combine=True".
output_dir = string(default=None)

# Filename of the simulation products manifest (JSON format)
manifest = string(default="products_manifest.json")


# Cosmological parameters
[cosmology]
# Hubble constant at z=0 [ km/s/Mpc ]
H0 = float(default=71.0, min=0.0)
# Density of non-relativistic matter in units of the critical density at z=0
OmegaM0 = float(default=0.27, min=0.0, max=1.0)


# Configurations for initialization/reconfiguration of the `logging` module
[logging]
# DEBUG:    Detailed information, typically of interest only when diagnosing
#           problems.
# INFO:     Confirmation that things are working as expected.
# WARNING:  An dinciation that something unexpected happended, or indicative
#           of some problem in the near future (e.g., "disk space low").
#           The software is still working as expected.
# ERROR:    Due to a more serious problem, the software has not been able to
#           perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable
#           to continue running.
level = option("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", default="INFO")

# Set the format of displayed messages
format = string(default="%(asctime)s [%(levelname)s] <%(name)s> %(message)s")

# Set the date/time format in messages
datefmt = string(default="%Y-%m-%dT%H:%M:%S")

# Set the logging filename (will create a `FileHandler`)
# If set to "" (empty string), then the `FileHandler` will be disabled.
filename = string(default="")
# Whether append messages to the above logging file instead of overwrite
appendmode = boolean(default=True)

# Set the stream used to initialize the `StreamHandler`
# If set to "" (empty string), then the `StreamHandler` will be disabled.
stream = option("stderr", "stdout", "", default="stderr")
