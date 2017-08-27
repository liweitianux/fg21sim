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
# - clusters: halos
# - pointsources


[extragalactic]

  # Extended emissions from the clusters of galaxies
  # The configurations in this ``[[clusters]]`` section may also be
  # used by the following ``[[halos]]`` section.
  [[clusters]]
  # The Press-Schechter formalism predicted halo distribution densities.
  ps_data = string(default=None)

  # Output the effective/inuse clusters catalog data (CSV file)
  catalog_outfile = string(default=None)

  # Output file for dumping the simulated cluster halos data in Python
  # native *pickle* format (i.e., .pkl)
  halos_dumpfile = string(default=None)

  # The fraction of the dark matter mass in galaxy clusters.
  f_darkmatter = float(default=0.8, min=0.5, max=1.0)

  # The minimum mass for clusters when to determine the galaxy clusters
  # total counts and their distributions.
  # Unit: [Msun]
  mass_min = float(default=2e14, min=1e12)

  # Minimum mass change of the main cluster to be regarded as a merger
  # event instead of an accretion event.
  # Unit: [Msun]
  merger_mass_min = float(default=1e12, min=1e10, max=1e14)

  # Mass ratio of the main and sub clusters, below which is regarded as
  # a major merger event.
  ratio_major = float(default=3.0, min=1.0, max=10.0)

  # The merger timescale, which roughly describes the duration of the
  # merger-induced disturbance (~2-3 Gyr).  This timescale is much longer
  # the merger crossing time (~1 Gyr), and is also longer than the lifetime
  # of radio halos.
  # Unit: [Gyr]
  tau_merger = float(default=3.0, min=1.0, max=5.0)

  # Magnetic field scaling relation for clusters
  # Reference: Cassano et al. 2012, A&A, 548, A100, Eq.(1)
  #
  # The mean magnetic field assumed
  # Unit: [uG]
  b_mean = float(default=1.9, min=0.1, max=10)
  # The index of the scaling relation
  b_index = float(default=1.5, min=0.0, max=3.0)

  # Filename prefix for this component
  prefix = string(default="cluster")
  # Output directory to save the simulated results
  output_dir = string(default=None)


  # Giant radio halos for clusters with recent major mergers
  [[halos]]
  # Roughly the fraction of turbulence energy transformed to accelerate
  # the electrons, describing the efficiency of turbulence acceleration.
  eta_turb = float(default=0.3, min=0.1, max=1.0)

  # Ratio of the total energy injected into cosmic-ray electrons during
  # the cluster life to its total thermal energy.
  eta_e = float(default=0.005, min=0.001, max=0.1)

  # Minimum and maximum Lorentz factor (i.e., energy) of the relativistic
  # electron spectrum.
  gamma_min = float(default=1e1)
  gamma_max = float(default=1e5)
  # Number of momentum points/cells for solving the Fokker-Planck
  # equation.
  gamma_np = integer(default=200, min=50)

  # Number of grid points used as the buffer region near the lower
  # boundary, and the value within this buffer region will be fixed to
  # avoid unphysical pile-up of low-energy electrons.
  # Reference: Donnert & Brunetti 2014, MNRAS, 443, 3564, Sec.(3.3)
  buffer_np = integer(default=5, min=0)

  # Time step for solving the Fokker-Planck equation
  # Unit: [Gyr]
  time_step = float(default=0.01, min=1e-5, max=0.1)

  # Electron injection, which is assumed to have a constant injection
  # rate and a power-law spectrum.
  injection_index = float(default=2.5, min=2.1, max=3.5)


  # Extragalactic point sources
  [[pointsources]]
  # Output directory to save the simulated catalog
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
