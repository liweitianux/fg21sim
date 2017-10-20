# Configurations for "fg21sim"
# -*- mode: conf -*-
#
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#
# This file contains the options corresponding the extragalactic emission
# components, which currently includes the following components:
# - clusters: halos
# - pointsources


[extragalactic]

  # Press-Schechter formalism to determine the dark matter halos
  # distribution with respect to masses and redshifts, from which
  # to further determine the total number of halos within a sky
  # patch and to sample the masses and redshifts for each halo.
  # NOTE: only consider the *dark matter* mass within the halo!
  [[psformalism]]
  # The model of the fitting function for halo mass distribution
  # For all models and more details:
  # https://hmf.readthedocs.io/en/latest/_autosummary/hmf.fitting_functions.html
  model = option("smt", "jenkins", "ps", default="smt")

  # The minimum (inclusive) and maximum (exclusive!) halo mass (dark
  # matter only) within which to calculate the halo mass distribution.
  # Unit: [Msun]
  M_min = float(default=1e12, min=1e10, max=1e14)
  M_max = float(default=1e16, min=1e14, max=1e18)
  # The logarithmic (base 10) step size for the halo masses; therefore
  # the number of intervals is: (log10(M_max) - log10(M_min)) / M_step
  M_step = float(default=0.01, min=0.001, max=0.1)

  # The minimum and maximum redshift within which to calculate the
  # halo mass distribution; as well as the step size.
  z_min = float(default=0.01, min=0.001, max=1.0)
  z_max = float(default=4.0, min=1.0, max=100)
  z_step = float(default=0.01, min=0.001, max=1.0)

  # Output file (NumPy ".npz" format) to save the calculated halo mass
  # distributions at every redshift.
  #
  # This file packs the following 3 NumPy arrays:
  # * ``dndlnm``:
  #   Shape: (len(z), len(mass))
  #   Differential mass function in terms of natural log of M.
  #   Unit: [Mpc^-3] (the little "h" is folded into the values)
  # * ``z``:
  #   Redshifts where the halo mass distribution is calculated.
  # * ``mass``:
  #   (Logarithmic-distributed) masses points.
  #   Unit: [Msun] (the little "h" is folded into the values)
  dndlnm_outfile = string(default=None)

  # Extended emissions from the clusters of galaxies
  # The configurations in this ``[[clusters]]`` section may also be
  # used by the following ``[[halos]]`` section.
  [[clusters]]
  # Output CSV file of the clusters catalog containing the simulated
  # mass, redshift, position, shape, and the recent major merger info.
  catalog_outfile = string(default=None)

  # Whether to directly use the (previously simulated) catalog file
  # specified as the above "catalog_outfile" option?
  # NOTE:
  # By using an existing catalog, the steps to derive these data are
  # simply skipped.
  # Due to the small number density of the galaxy clusters, the simulated
  # results within a small patch of sky (e.g., 100 [deg^2]) show
  # significant fluctuations (several or even several tens of times
  # of differences between simulations).  Therefore, one may run many
  # tests and only create images at some frequencies necessary for
  # testing, then select the satisfying one to continue the simulation
  # to generate images at all frequencies.
  use_output_catalog = boolean(default=False)

  # Output CSV file of the halos catalog containing the calculated
  # properties of the simulated halos.
  halos_catalog_outfile = string(default=None)

  # Whether to dump the whole data of the simulated halos in Python
  # native pickle format (i.e., ".pkl") to a file with the same basename
  # as the above ``halos_catalog_outfile``?
  # The dumped data also includes the derived electron spectrum for
  # each halo, therefore this file can be reloaded back in order to
  # calculate the emissions at other frequencies.
  dump_halos_data = boolean(default=True)

  # Whether to directly use the (previously dumped) halos data (".pkl")
  # as specified by the above ``halos_catalog_outfile`` and
  # ``dump_halos_data`` options?
  # In this way, the radio emissions at additional frequencies can be
  # easily (and consistently) calculated.
  use_dump_halos_data = boolean(default=False)

  # The minimum mass for clusters when to determine the galaxy clusters
  # total counts and their distributions.
  # Unit: [Msun]
  mass_min = float(default=1e14, min=1e12)

  # Boost the number of expected cluster number within the sky coverage
  # by the specified times.
  # (NOTE: mainly for testing purpose.)
  boost = float(default=1.0, min=0.1, max=1e4)

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
  tau_merger = float(default=2.0, min=1.0, max=5.0)

  # The fraction of the magnetic field energy density w.r.t. the ICM
  # thermal energy density, which is used to determine the mean magnetic
  # field strength within the ICM and is also assumed to be uniform.
  eta_b = float(default=0.001, min=1e-5, max=0.1)

  # Filename prefix for this component
  prefix = string(default="cluster")
  # Output directory to save the simulated results
  output_dir = string(default=None)


  # Giant radio halos for clusters with recent major mergers
  [[halos]]
  # Roughly the fraction of turbulence energy transformed to accelerate
  # the electrons, describing the efficiency of turbulence acceleration.
  eta_turb = float(default=0.2, min=0.1, max=1.0)

  # Ratio of the total energy injected into cosmic-ray electrons during
  # the cluster life to its total thermal energy.
  eta_e = float(default=0.003, min=0.001, max=0.1)

  # Minimum and maximum Lorentz factor (i.e., energy) of the relativistic
  # electron spectrum.
  gamma_min = float(default=1e1)
  gamma_max = float(default=1e5)
  # Number of momentum points/cells for solving the Fokker-Planck
  # equation.
  gamma_np = integer(default=200, min=100)

  # Number of cells used as the buffer regions near both the lower
  # and upper boundaries, and the value within the buffer regions will
  # be fixed to avoid unphysical pile-ups.
  # It is suggested to be about 5%-10% of the above ``gamma_np``.
  buffer_np = integer(default=10, min=0)

  # Time step for solving the Fokker-Planck equation
  # Unit: [Gyr]
  time_step = float(default=0.02, min=1e-4, max=0.1)

  # Electron injection, which is assumed to have a constant injection
  # rate and a power-law spectrum.
  injection_index = float(default=2.4, min=2.1, max=3.5)


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
