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
# This file contains the options corresponding the extragalactic emission
# components, which currently includes the following components:
# - clusters
#
# NOTE:
# - The input templates for simulations should be HEALPix full-sky maps.
# - The input catalog should be in CSV format.


[extragalactic]

  # Emissions from the clusters of galaxies
  [[clusters]]
  # The clusters catalog derived from the Hubble Volume Project (CSV file)
  catalog = string(default=None)
  # Output the effective/inuse clusters catalog data (CSV file)
  catalog_outfile = string(default=None)

  # The fraction that a cluster hosts a radio halo
  halo_fraction = float(default=None)

  # Resolution (unit: arcmin) for simulating each cluster, which are finally
  # mapped to the HEALPix map of Nside specified in "[common]" section.
  resolution = float(default=0.5)

  # Filename prefix for this component
  prefix = string(default="egcluster")
  # Whether save this component to disk
  save = boolean(default=True)
  # Output directory to save the simulated results
  output_dir = string(default=None)

  # Emission from various types of point sources (PS)
  [[pointsources]]
  # Enabled types of PS to be simulated
  ps_types = string_list(default=list("starforming", "starbursting", "radioquiet", "fr1", "fr2"))

  # FIXME:
  # The number of each PS type should be determined from the observation
  # or theoretical prediction, e.g., luminosity function.
  # On the other hand, due to the large number of PS, an option
  # "number_fraction" can be supplied to specify the fraction of total
  # PS number to be only simulated, which may be useful for testing purpose.
  #
  # number_fraction = float(default=1.0)

  # Resolution (unit: arcmin) of the simulation grid for each PS, which are
  # finally mapped to the HEALPix map of resolution "nside".
  resolution = float(default=0.5)

  # FIXME:
  # Move this option to "[output]" section;
  # Rename "filename_pattern" in "[output]" section to "hpmap_pattern" ?
  #
  # Filename pattern for the simulated catalogs of each PS type, which will
  # be saved in CSV format.
  catalog_pattern = "catalog_{prefix}.csv"

  # Filename prefix (with additional prefix specified for each PS type)
  prefix = string(default="egps")
  # Whether save the *combined maps* of all enabled PS types to disk
  save = boolean(default=True)
  # Output directory to save the simulated maps and catalogs
  output_dir = string(default=None)

    # PS type: Star-forming galaxies
    [[[starforming]]]
    # Number of point sources of this PS type
    number = integer(default=1000)
    # Additional filename prefix to identify this PS type, which will be
    # *appended* to the upper-level "prefix" of "[pointsources]" section.
    prefix2 = string(default="sf")
    # Whether save the simulated maps of this PS types to disk
    save2 = boolean(default=False)

    # PS type: Star-bursting galaxies
    [[[starbursting]]]
    number = integer(default=1000)
    prefix2 = string(default="sb")
    save2 = boolean(default=False)

    # PS type: radio-quiet AGNs
    [[[radioquiet]]]
    number = integer(default=1000)
    prefix2 = string(default="rq")
    save2 = boolean(default=False)

    # PS type: radio-loud AGNs (FR I)
    [[[fr1]]]
    number = integer(default=1000)
    prefix2 = string(default="fr1")
    save2 = boolean(default=False)

    # PS type: radio-loud AGNs (FR II)
    [[[fr2]]]
    number = integer(default=1000)
    prefix2 = string(default="fr2")
    save2 = boolean(default=False)
