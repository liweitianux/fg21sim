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
# - halos
# - pointsources
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
  halo_fraction = float(default=None, min=0.0, max=1.0)

  # Resolution (unit: arcmin) for simulating each cluster, which are finally
  # mapped to the HEALPix map of Nside specified in "[common]" section.
  resolution = float(default=0.5, min=0.0)

  # Filename prefix for this component
  prefix = string(default="egcluster")
  # Whether save this component to disk
  save = boolean(default=True)
  # Output directory to save the simulated results
  output_dir = string(default=None)


  # Emission of giant radio halos from galaxy clusters
  [[halos]]
  # Maximum redshift until where to tracing the cluster merging history
  # (e.g., when calculating the electron spectrum)
  zmax = float(default=3.0, min=0.0)

  # Mass threshold of the sub-cluster to be regarded as a significant
  # merger. (unit: Msun)
  merger_mass_th = float(default=1e13, min=1e12)

  # Minimum mass change of the main-cluster to be regarded as a merger
  # event rather than accretion. (unit: Msun)
  merger_mass_min = float(default=1e12, min=1e10)

  # Radius of the giant radio halo in clusters (unit: kpc)
  # XXX: currently only support a constant radius of halos
  radius = float(default=500, min=100)

  # Magnetic field assumed for the cluster (unit: uG)
  # XXX: currently only support a constant magnetic field
  magnetic_field = float(default=0.5, min=0.1, max=10)

  # Fraction of the turbulence energy in the form of magneto-sonic waves.
  eta_t = float(default=0.3, min=0.0, max=1.0)

  # Ratio of the total energy injected in cosmic-ray electrons during the
  # cluster life to the present-day total thermal energy of the cluster.
  eta_e = float(default=0.003, min=0.0, max=0.1)

  # Minimum and maximum Lorentz factor (i.e., energy) of the relativistic
  # electron spectrum.
  pmin = float(default=1e1)
  pmax = float(default=1e5)

  # Number of points for the grid used during solving the Fokker-Planck
  # equation to calculate the electron spectrum.
  pgrid_num = integer(default=100, min=10)

  # Number of grid points used as the buffer region near the lower
  # boundary, and the value within this buffer region will be fixed to
  # avoid unphysical pile-up of low-energy electrons.
  buffer_np = integer(default=5, min=0)

  # Time step for solving the Fokker-Planck equation (unit: Gyr)
  time_step = float(default=0.01, min=1e-5, max=1.0)

  # Index of the power-law spectrum assumed for the injected electrons.
  injection_index = float(default=2.5)


  # Extragalactic point sources
  [[pointsources]]
  # Whether save this point source catelogue to disk
  save = boolean(default=True)
  # Output directory to save the simulated catelogues
  output_dir = string(default="PS_tables")
  # PS components to be simulated
  pscomponents = string_list(default=list())
  # Resolution [arcmin]
  resolution = float(default=0.6, min=0.0)

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
