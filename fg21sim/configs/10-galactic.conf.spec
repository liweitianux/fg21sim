# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# This file contains the general configurations, which control the general
# behaviors, or will be used in other configuration sections.
#
# NOTE:
# The input templates for simulations should be HEALPix full-sky maps.
#


[galactic]

  # Synchrotron emission component (unpolarized)
  [[synchrotron]]
  # The template map for the simulation.
  template = string(default=None)
  # The frequency of the template map (same unit as in [frequency] section)
  template_freq = float(default=None)
  # The unit of the template map pixel
  template_unit = string(default=None)

  # Spectral index map
  indexmap = string(default=None)

  # Whether add fluctuations on the small scales
  add_smallscales = boolean(default=True)

  # Filename prefix for this component
  prefix = string(default="gsync")
  # Whether save this component to disk
  save = boolean(default=True)
  # Output directory to save the simulated results
  output_dir = string(default=None)

  # Free-free bremsstrahlung emission component
  [[freefree]]
  # The H{\alpha} map used as the free-free emission template
  halphamap = string(default=None)
  # The unit of the H{\alpha} template (e.g., "Rayleigh")
  halphamap_unit = string(default=None)

  # The 100-{\mu}m dust map used for dust absorption correction
  dustmap = string(default=None)
  # The unit of the above dust map (e.g., "MJy/sr")
  dustmap_unit = string(default=None)

  # Filename prefix for this component
  prefix = string(default="gfree")
  # Whether save this component to disk
  save = boolean(default=True)
  # Output directory to save the simulated results
  output_dir = string(default=None)

  # Supernova remnants emission
  [[snr]]
  # The Galactic SNRs catalog data (CSV file)
  catalog = string(default=None)
  # Output the effective/inuse SNRs catalog data (CSV file)
  catalog_outfile = string(default=None)

  # Resolution (unit: arcmin) for simulating each SNR, which are finally
  # mapped to the HEALPix map of Nside specified in "[common]" section.
  resolution = float(default=1.0)

  # Filename prefix for this component
  prefix = string(default="gsnr")
  # Whether save this component to disk
  save = boolean(default=True)
  # Output directory to save the simulated results
  output_dir = string(default=None)
