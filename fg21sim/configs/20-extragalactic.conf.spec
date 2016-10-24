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

  # Extragalactic point sources
  [[pointsources]]
  # Whether save this point source catelogue to disk
  save = boolean(default=True)
  # Output directory to save the simulated catelogues
  output_dir = string(default="PS_tables")
  # PS components to be simulated
  pscomponents=string_list(default=list())
  # Number of each type of point source
  # Star forming
  [[[starforming]]]
  # Number of samples
  numps = integer(default=1000)
  # Prefix
  prefix = string(default="SF")

  [[[starbursting]]]
  # Number of samples
  numps = integer(default=1000)
  # Prefix
  prefix = string(default="SB")

  [[[radioquiet]]]
  # Number of samples
  numps = integer(default=1000)
  # Prefix
  prefix = string(default="RQ")

  [[[FRI]]]
  # Number of samples
  numps = integer(default=1000)
  # Prefix
  prefix = string(default="FRI")

  [[[FRII]]]
  # Number of samples
  numps = integer(default=1000)
  # Prefix
  prefix = string(default="FRII")
