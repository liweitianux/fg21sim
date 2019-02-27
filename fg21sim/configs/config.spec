#
# Configurations for "fg21sim"
# Syntax: `ConfigObj`, https://github.com/DiffSK/configobj
#


# Foreground components to be simulated
[foregrounds]
# Diffuse Galactic synchrotron emission (unpolarized)
galactic/synchrotron = boolean(default=False)
# Diffuse Galactic free-free emission
galactic/freefree = boolean(default=False)
# Galactic supernova remnants emission
galactic/snr = boolean(default=False)
# Extragalactic clusters of galaxies emission
extragalactic/clusters = boolean(default=False)


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
  xcenter = float(default=0, min=0, max=360)
  ycenter = float(default=-27, min=-90, max=90)

  # The image dimensions (i.e., number of pixels) of the sky patch,
  # along the X (R.A./longitude) and Y (Dec./latitude) axes.
  # Default: 1800x1800 => 10x10 [deg^2] (20 arcsec/pixel)
  xsize = integer(default=1800, min=1)
  ysize = integer(default=1800, min=1)

  # Pixel size [arcsec]
  pixelsize = float(default=20, min=0)

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
start = float(default=None)
stop = float(default=None)
step = float(default=None)


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


#
# Galactic emission components
#

[galactic]

  #
  # Synchrotron emission component (unpolarized)
  #
  [[synchrotron]]
  # The template map for the simulation, e.g., Haslam 408 MHz survey.
  # Unit: [K] (Kelvin)
  template = string(default=None)
  # The frequency of the template map.
  # Unit: [MHz]
  template_freq = float(default=None)

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


  #
  # Free-free bremsstrahlung emission component
  #
  [[freefree]]
  # The Hα map from which to derive the free-free emission
  # Unit: [Rayleigh]
  halphamap = string(default=None)

  # The 100-μm dust map used to correct Hα dust absorption
  # Unit: [MJy/sr]
  dustmap = string(default=None)

  # Effective dust fraction in the LoS actually absorbing Halpha
  dust_fraction = float(default=0.33, min=0, max=1)

  # Halpha absorption threshold:
  # When the dust absorption goes rather large, the true Halpha
  # absorption can not well determined.  This configuration sets the
  # threshold below which the dust absorption can be well determined,
  # while the sky regions with higher absorption are masked out due
  # to unreliable absorption correction.
  # Unit: [mag]
  halpha_abs_th = float(default=1)

  # The electron temperature assumed for the ionized interstellar medium
  # that generating Hα emission.
  # Unit: [K]
  electron_temperature = float(default=7000)

  # Filename prefix for this component
  prefix = string(default="gfree")
  # Output directory to save the simulated results
  output_dir = string(default=None)


  #
  # Supernova remnants emission
  #
  [[snr]]
  # The Galactic SNRs catalog data (CSV file)
  catalog = string(default=None)
  # Output the effective/inuse SNRs catalog data (CSV file)
  catalog_outfile = string(default=None)

  # Resolution for simulating each SNR template, which are finally
  # mapped to the all-sky HEALPix map if used.
  # Unit: [arcsec]
  resolution = float(default=30)

  # Filename prefix for this component
  prefix = string(default="gsnr")
  # Output directory to save the simulated results
  output_dir = string(default=None)


#
# Extragalactic emission components
#

[extragalactic]
  #
  # Press-Schechter formalism to determine the cluster distributions
  # with respect to mass and redshift, from which to further determine
  # the total number of clusters within a sky patch and to sample the
  # masses and redshifts for each cluster.
  #
  [[psformalism]]
  # The model of the fitting function for halo/cluster mass distribution
  # For all models and more details:
  # https://hmf.readthedocs.io/en/latest/_autosummary/hmf.fitting_functions.html
  model = option("smt", "jenkins", "ps", default="ps")

  # The minimum (inclusive) and maximum (exclusive!) cluster mass
  # within which to calculate the halo mass distribution.
  # Unit: [Msun]
  M_min = float(default=1e12, min=1e10, max=1e14)
  M_max = float(default=1e16, min=1e14, max=1e18)
  # The 10-based logarithmic step size for the halo masses; therefore
  # the number of intervals is: (log10(M_max) - log10(M_min)) / M_step
  M_step = float(default=0.01, min=0.001, max=0.1)

  # The minimum and maximum redshift within which to calculate the
  # halo mass distribution; as well as the step size.
  z_min = float(default=0.01, min=0.001, max=1)
  z_max = float(default=4, min=1, max=100)
  z_step = float(default=0.01, min=0.001, max=1)

  # Output file (NumPy ".npz" format) to save the calculated halo mass
  # distributions at every redshift.
  #
  # This file packs the following 3 NumPy arrays:
  # * ``z``:
  #   Redshifts where the halo mass distribution is calculated.
  # * ``mass``:
  #   (Logarithmic-distributed) mass points.
  #   Unit: [Msun] (NOTE: the little "h" is folded into the values)
  # * ``dndlnm``:
  #   Shape: (len(z), len(mass))
  #   Differential mass function in terms of ln(M).
  #   Unit: [Mpc^-3] (NOTE: the little "h" is folded into the values)
  dndlnm_outfile = string(default="dndlnm.npz")


  #
  # Extended emissions from the clusters of galaxies
  # The configurations in this ``[[clusters]]`` section may also be
  # used by the following ``[[halos]]`` section.
  #
  [[clusters]]
  # Output CSV file of the cluster catalog containing the simulated
  # mass, redshift, position, shape, recent merger info, etc.
  catalog_outfile = string(default="cluster.catalog.csv")

  # Whether to dump the raw data of the simulated cluster catalog in
  # Python native pickle format (i.e., ".pkl") to a file with the same
  # basename as the above ``catalog_outfile``?
  # The dumped data can be easily loaded back for reuse.
  dump_catalog_data = boolean(default=True)

  # Whether to directly use the (previously simulated) catalog data as
  # specified by the above "catalog_outfile" and ``dump_catalog_data``
  # options?
  #
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
  use_dump_catalog_data = boolean(default=False)

  # Output CSV file of the halos catalog containing the calculated
  # properties of the simulated halos.
  halos_catalog_outfile = string(default="cluster.halos.csv")

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
  mass_min = float(default=1e14, min=1e13)

  # Boost the number of expected cluster number within the sky coverage
  # by the specified times.
  # NOTE: for testing usage.
  boost = float(default=1)

  # Minimal elongated fraction for creating the images of radio halos
  # The ``felong`` is defined as ``felong = b/a``, similar to the Hubble
  # classification for the elliptical galaxies.  ``felong_min = 1.0``
  # means no elongation, and ``felong_min = 0.6`` is a good choice as
  # the observed radio halos are generally regular.
  felong_min = float(default=1, min=0.1, max=1)

  # Number of most powerful halos to be dropped out.
  halo_dropout = integer(default=0, min=0)

  # Minimum mass change of the main cluster to be regarded as a merger
  # event instead of an accretion event.
  # Unit: [Msun]
  merger_mass_min = float(default=1e13, min=1e11, max=1e14)

  # The trace back time when to stop tracing the merging history of
  # clusters.  ~2-3 Gyr should be enough since the turbulence acceleration
  # effective time ~<1 Gyr and the halo lifetime is also short compared
  # to mergers.
  # Unit: [Gyr]
  time_traceback = float(default=3, min=1, max=5)

  # The temperature of the outer gas surrounding the cluster.  Accretion
  # shocks form near the cluster virial radius during the cluster formation,
  # which can heat the cluster ICM to have a higher temperature than the
  # virial temperature:
  #     kT_icm ~ kT_vir + 1.5 * kT_out,
  # with: kT_out ~ 0.5 [keV]
  # Reference: Fujita et al. 2003, ApJ, 584, 190; Eq.(49)
  # Unit: [keV]
  kT_out = float(default=0.5, min=0)

  # Whether to make the simulated sky maps?  It is useful to disable the
  # map generation during the parameter tuning.
  make_maps = boolean(default=True)

  # Filename prefix for this component
  prefix = string(default="cluster")
  # Output directory to save the simulated results
  output_dir = string(default=None)


  #
  # Giant radio halos
  #
  [[halos]]
  # The fraction of the thermal energy injected into the cosmic-ray
  # electrons during the cluster life time.
  eta_e = float(default=0.001, min=0, max=1)
  # The spectral index of the injected primary electrons.
  injection_index = float(default=2.3, min=2, max=3)

  # The fraction of merger energy transferred into the turbulence.
  eta_turb = float(default=0.15, min=0, max=1)

  # The base energy fraction of the turbulence to the ICM thermal energy
  # (for a relaxed system).
  # x_turb ~< 5% [Vazza et al. 2011, A&A, 529, A17]
  x_turb = float(default=0.01, min=0, max=0.5)

  # A custom factor to tune the turbulent acceleration efficiency.
  # NOTE: This parameter incorporates the efficiency factor describing
  #       the effectiveness of the ICM plasma instabilities.
  f_acc = float(default=0.1, min=0.1, max=10)

  # The energy density ratio of cosmic ray to the thermal ICM.
  # NOTE: Equipartition between the magnetic field and cosmic ray is
  #       assumed, i.e., eta_b == x_cr.
  x_cr = float(default=0.015, min=0, max=1)

  # The scaling index of the diffusion coefficient (D_γγ) w.r.t. the
  # mass of the cluster.
  mass_index = float(default=0, min=0, max=3)

  # The factor that is multiplied to the turbulence injection radius
  # to derive the radio halo radius.
  f_radius = float(default=0.7, min=0.1, max=10)
  # The scaling index of the halo radius (R_halo) w.r.t. the virial
  # radius of the cluster.
  radius_index = float(default=1.7, min=0, max=3)

  # Minimum and maximum Lorentz factor (i.e., energy) of the relativistic
  # electron spectrum.
  gamma_min = float(default=1)
  gamma_max = float(default=1e6)
  # Number of cells on the logarithmic momentum grid used to solve the
  # Fokker-Planck equation.
  gamma_np = integer(default=256)

  # Number of cells used as the buffer regions near both the lower
  # and upper boundaries, within which the values will be replaced by
  # extrapolating from the inner-region data, in order to avoid the
  # unphysical particle pile-ups.
  #
  # NOTE: To disable the boundary fix, set this to 0, otherwise, set to
  #       a number >= 2.  It is suggested to be about 5%-10% of the
  #       ``gamma_np``.
  buffer_np = integer(default=10, min=0)

  # Time step for solving the Fokker-Planck equation
  # Unit: [Gyr]
  time_step = float(default=0.02, min=1e-4, max=0.1)

  # How long the period before the merger begins, which is used to derive
  # an approximately steady initial electron spectrum.  During this period,
  # the acceleration is turned off and only leaves energy loss mechanisms.
  # Unit: [Gyr]
  time_init = float(default=1, min=0)

  # Parameters of the beta-model that is used to describe the gas density
  # profile of the cluster.
  # The fraction of the core radius to cluster's virial radius.
  f_rc = float(default=0.1)
  # The slope parameter (i.e., beta).
  beta = float(default=0.6667)

  # The emissivity acceleration factor and spectral index thresholds for
  # determining whether the halo is genuine/formed.
  genuine_emfacc_th = float(default=100)
  genuine_index_th = float(default=3.5)
