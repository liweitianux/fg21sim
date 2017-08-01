# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# This file contains the general configurations, which control the general
# behaviors, or will be used in other configuration sections.


# Foreground components to be simulated
[foregrounds]
# Diffuse Galactic synchrotron emission (unpolarized)
galactic/synchrotron = boolean(default=False)

# Diffuse Galactic free-free emission
galactic/freefree = boolean(default=False)

# Galactic supernova remnants emission
galactic/snr = boolean(default=False)

#  Extragalactic clusters of galaxies emission
extragalactic/clusters = boolean(default=False)

# Emission from multiple types of extragalactic point sources
# NOTE: This component is not well integrated and tested at the moment
extragalactic/pointsources = boolean(default=False)


# Simulation sky/region configurations
[sky]
# Type of the input/output simulation sky
# + patch:
#       Input/output sky template is only a (square) patch of the sky.
#       The simulated output maps have the same coverage/field as the
#       input template, as well as the coordinate projection.
# + healpix:
#       Input/output sky template covers (almost) all sky, and stored
#       in HEALPix format.  The simulated output maps will also be
#       all-sky using the HEALPix projection.
type = option("patch", "healpix", default="patch")

  # Configurations for input/output sky patch
  [[patch]]
  # The (R.A., Dec.) coordinate of the sky patch center [deg]
  xcenter = float(default=0.0, min=0.0, max=360.0)
  ycenter = float(default=0.0, min=-90.0, max=90.0)

  # The image dimensions (i.e., number of pixels) of the sky patch,
  # along the X (R.A./longitude) and Y (Dec./latitude) axes.
  # Default: 1200x1200 => 10x10 [deg^2] (30 arcsec/pixel)
  xsize = integer(default=1200, min=1)
  ysize = integer(default=1200, min=1)

  # Pixel size [arcsec]
  pixelsize = float(default=30.0, min=0.0)

  # Configurations for input/output HEALPix sky
  [[healpix]]
  # HEALPix Nside value, i.e., pixel resolution
  nside = integer(default=1024, min=128)


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

# Filename pattern for the output products, which will be finally
# formatted using `str.format()`.
filename_pattern = string(default="{prefix}_{frequency:06.2f}.fits")

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

# Filename of the simulation products manifest (JSON format), which
# records all output products together with their sizes and MD5 hashes.
# Do not create such a manifest if this option is not specified.
manifest = string(default=None)


# Cosmological parameters
# References: Komatsu et al. 2011, ApJS, 192, 18; Tab.(1)
[cosmology]
# Hubble constant at z=0 [ km/s/Mpc ]
H0 = float(default=71.0, min=0.0)
# Density of non-relativistic matter in units of the critical density at z=0
OmegaM0 = float(default=0.27, min=0.0, max=1.0)
# Density of the baryon at present day
Omegab0 = float(default=0.046, min=0.0, max=1.0)
# Present-day rms density fluctuations on a scale of 8 h^-1 Mpc
sigma8 = float(default=0.81, min=0.0)


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
