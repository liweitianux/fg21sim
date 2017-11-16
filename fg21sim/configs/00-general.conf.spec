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
  # The (R.A., Dec.) coordinate of the sky patch center
  # Unit: [deg]
  # (MWA EoR0 field center: (0, -27))
  xcenter = float(default=0.0, min=0.0, max=360.0)
  ycenter = float(default=-27.0, min=-90.0, max=90.0)

  # The image dimensions (i.e., number of pixels) of the sky patch,
  # along the X (R.A./longitude) and Y (Dec./latitude) axes.
  # Default: 1800x1800 => 10x10 [deg^2] (20 arcsec/pixel)
  xsize = integer(default=1800, min=1)
  ysize = integer(default=1800, min=1)

  # Pixel size [arcsec]
  pixelsize = float(default=20.0, min=0.0)

  # Configurations for input/output HEALPix sky
  [[healpix]]
  # HEALPix Nside value, i.e., pixel resolution
  nside = integer(default=1024, min=128)


# Frequencies specification of the simulation products
[frequency]
# How to specify the frequencies
# + custom:
#       directly specify the frequency values using the "frequencies" config
# + calc:
#       calculate the frequency values by "start", "stop", and "step"
type = option("custom", "calc", default="custom")

# The frequency values to be simulated if above "type" is "custom".
# Unit: [MHz]
frequencies = float_list(default=list())

# Parameters to calculate the frequencies
# NOTE: "start" and "stop" frequencies are both inclusive.
# Unit: [MHz]
start = float(default=None, min=0.0)
stop = float(default=None, min=0.0)
step = float(default=None, min=0.0)


# Configuration for output products
[output]
# Filename pattern for the output products, which will be finally
# formatted using `str.format()`.
filename_pattern = string(default="{prefix}_{frequency:06.2f}.fits")

# Use single-precision float instead of double (to save spaces)
float32 = boolean(default=True)

# Whether to calculate the checksum for the output FITS file?
# NOTE: May cost significantly more time on writing FITS file.
checksum = boolean(default=False)

# Whether to overwrite existing files (e.g., maps, catalogs, manifest, ...)
clobber = boolean(default=False)

# Filename of the simulation products manifest (JSON format), which
# records all output products together with their sizes and MD5 hashes.
# Do not create such a manifest if this option is not specified.
manifest = string(default=None)


# Cosmological parameters
# References: Komatsu et al. 2011, ApJS, 192, 18; Tab.(1)
[cosmology]
# Hubble constant at z=0; [km/s/Mpc]
H0 = float(default=71.0, min=0.0)
# Density of non-relativistic matter in units of the critical density at z=0
OmegaM0 = float(default=0.27, min=0.0, max=1.0)
# Density of the baryon at present day
Omegab0 = float(default=0.046, min=0.0, max=1.0)
# Present-day CMB temperature; [K]
Tcmb0 = float(default=2.725)
# Present-day rms density fluctuations on a scale of 8 h^-1 [Mpc]
sigma8 = float(default=0.81, min=0.0)
# Scalar spectral index
ns = float(default=0.96, min=0.0)


# Configurations for initialization/reconfiguration of the `logging` module
[logging]
# debug:    Detailed information, typically of interest only when diagnosing
#           problems.
# info:     Confirmation that things are working as expected.
# warning:  An indication that something unexpected happended, or indicative
#           of some problem in the near future (e.g., "disk space low").
#           The software is still working as expected.
# error:    Due to a more serious problem, the software has not been able to
#           perform some function.
# critical: A serious error, indicating that the program itself may be unable
#           to continue running.
level = option("debug", "info", "warning", "error", "critical", default="info")

# Set the format of displayed messages
format = string(default="%(asctime)s [%(levelname)s] <%(name)s:%(lineno)d> %(message)s")

# Set the date/time format in messages
datefmt = string(default="%H:%M:%S")

# Set the logging filename (will create a `FileHandler`)
# If set to "" (empty string), then the `FileHandler` will be disabled.
filename = string(default="")

# Set the stream used to initialize the `StreamHandler`
# If set to "" (empty string), then the `StreamHandler` will be disabled.
stream = option("stderr", "stdout", "", default="stderr")
