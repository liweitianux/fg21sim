# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# There are various configuration options that specify the input data
# and/or templates required by the simulations, the properties of the input
# data, the output products, as well as some parameters affecting the
# simulation behaviors.
#
# This file contains the options corresponding the Galactic emission
# components, which currently includes the following components:
# - synchrotron
# - freefree
# - snr
#
# NOTE:
# - The input templates for simulations should be HEALPix full-sky maps.
# - The input catalog should be in CSV format.


[galactic]

  # Synchrotron emission component (unpolarized)
  [[synchrotron]]
  # The template map for the simulation, e.g., Haslam 408 MHz survey.
  # Unit: [K] (Kelvin)
  template = string(default=None)
  # The frequency of the template map.
  # Unit: [MHz]
  template_freq = float(default=None, min=0.0)

  # Spectral index map
  indexmap = string(default=None)

  # Whether add fluctuations on the small scales according the angular
  # power spectrum prediction?
  add_smallscales = boolean(default=False)
  # Range of multipole moments (l) of the angular power spectrum.
  # The power spectrum will be cut off to a constant for multipole l < lmin.
  # NOTE: Update the ``lmax`` accordingly w.r.t. ``sky/healpix/nside``.
  #       Generally, lmax = 3 * nside - 1
  lmin = integer(min=0, default=10)
  lmax = integer(min=1, default=3071)

  # Filename prefix for this component
  prefix = string(default="gsync")
  # Output directory to save the simulated results
  output_dir = string(default=None)

  # Free-free bremsstrahlung emission component
  [[freefree]]
  # The Hα map from which to derive the free-free emission
  # Unit: [Rayleigh]
  halphamap = string(default=None)

  # The 100-μm dust map used to correct Hα dust absorption
  # Unit: [MJy/sr]
  dustmap = string(default=None)

  # Effective dust fraction in the LoS actually absorbing Halpha
  dust_fraction = float(default=0.33, min=0.1, max=1.0)

  # Halpha absorption threshold:
  # When the dust absorption goes rather large, the true Halpha
  # absorption can not well determined.  This configuration sets the
  # threshold below which the dust absorption can be well determined,
  # while the sky regions with higher absorption are masked out due
  # to unreliable absorption correction.
  # Unit: [mag]
  halpha_abs_th = float(default=1.0)

  # The electron temperature assumed for the ionized interstellar medium
  # that generating H{\alpha} emission.
  # Unit: [K]
  electron_temperature = float(default=7000.0, min=1000)

  # Filename prefix for this component
  prefix = string(default="gfree")
  # Output directory to save the simulated results
  output_dir = string(default=None)

  # Supernova remnants emission
  [[snr]]
  # The Galactic SNRs catalog data (CSV file)
  catalog = string(default=None)
  # Output the effective/inuse SNRs catalog data (CSV file)
  catalog_outfile = string(default=None)

  # Resolution for simulating each SNR template, which are finally
  # mapped to the all-sky HEALPix map if used.
  # Unit: [arcsec]
  resolution = float(default=30.0, min=5.0)

  # Filename prefix for this component
  prefix = string(default="gsnr")
  # Output directory to save the simulated results
  output_dir = string(default=None)
