# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Galactic supernova remnants (SNRs) emission simulations.

References
----------
.. [Jelic2008]
   Jelić, V. et al.,
   "Foreground simulations for the LOFAR-epoch of reionization experiment",
   2008, MNRAS, 389, 1319-1335,
   http://adsabs.harvard.edu/abs/2008MNRAS.389.1319J

.. [Green2014]
   Green, D. A.,
   "A catalogue of 294 Galactic supernova remnants",
   2014, Bulletin of the Astronomical Society of India, 42, 47-58,
   http://adsabs.harvard.edu/abs/2014BASI...42...47G

.. [GreenSNRDataWeb]
   A Catalogue of Galactic Supernova Remnants
   http://www.mrao.cam.ac.uk/surveys/snrs/
"""

import os
import logging

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from ..sky import get_sky
from ..utils.wcs import make_wcs
from ..utils.convert import Fnu_to_Tb
from ..utils.grid import make_ellipse
from ..utils.units import UnitConversions as AUC


logger = logging.getLogger(__name__)


class SuperNovaRemnants:
    """
    Simulate the Galactic supernova remnants emission.

    The simulation follows the method adopted by [Jelic2008]_, which is
    based on the Galactic SNRs catalog maintained by *D. A. Green*
    [Green2014]_ and [GreenSNRDataWeb]_, which contains 294 SNRs (2014-May).
    However, some SNRs have incomplete data which are excluded, while
    some SNRs with uncertain properties are currently kept.

    Every SNR is simulated as an *ellipse* of *uniform brightness* on
    a local coordinate grid with relatively higher resolution compared
    to the output HEALPix map, which is then mapped to the output HEALPix
    map by down-sampling.

    Parameters
    ----------
    configs : `ConfigManager`
        A `ConfigManager` instance containing default and user configurations.
        For more details, see the example configuration specifications.

    Attributes
    ----------
    TODO
    """
    # Component name
    compID = "galactic/snr"
    name = "Galactic SNRs"

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()

        self.sky = get_sky(configs)
        self.sky.add_header("CompID", self.compID, "Emission component ID")
        self.sky.add_header("CompName", self.name, "Emission component")
        self.sky.add_header("BUNIT", "K", "[Kelvin] Data unit")
        self.sky.creator = __name__

    def _set_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp = self.compID
        self.catalog_path = self.configs.get_path(comp+"/catalog")
        self.catalog_outfile = self.configs.get_path(comp+"/catalog_outfile")
        self.resolution = self.configs.getn(comp+"/resolution")  # [ arcsec ]
        self.prefix = self.configs.getn(comp+"/prefix")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.clobber = self.configs.getn("output/clobber")
        self.frequencies = self.configs.frequencies  # [MHz]
        logger.info("Loaded and set up configurations")

    def _load_catalog(self):
        """
        Load the Galactic SNRs catalog data.

        Catalog columns:
        * glon, glat : SNR coordinate, Galactic coordinate, [deg]
        * size_major, size_minor : SNR angular sizes; major and minor axes
            of the ellipse fitted to the SNR, or the diameter of the fitted
            circle if the SNR is nearly circular;
            originally in [arcmin], converted to [arcsec]
        * flux : Flux density at 1 GHz, [Jy]
        """
        self.catalog = pd.read_csv(self.catalog_path)
        nrow, ncol = self.catalog.shape
        logger.info("Loaded SNRs catalog data from: {0}".format(
            self.catalog_path))
        logger.info("SNRs catalog data: {0} objects, {1} columns".format(
            nrow, ncol))
        # The flux densities are given at 1 [GHz] -> 1000 [MHz]
        self.catalog_flux_freq = 1e3  # [MHz]
        # Convert ``size_major`` and ``size_minor`` from unit [arcmin]
        # to [arcsec]
        self.catalog["size_major"] *= AUC.arcmin2arcsec
        self.catalog["size_minor"] *= AUC.arcmin2arcsec

    def _save_catalog_inuse(self):
        """
        Save the effective/inuse SNRs catalog data to a CSV file.

        NOTE
        ----
        - Only the effective/inuse SNRs are saved (i.e., without the ones
          that are filtered out).
        - Also save the simulated rotation column.
        - The unnecessary columns are striped.
        """
        if self.catalog_outfile is None:
            logger.warning("Catalog output file not set; skipped!")
            return
        # Create directory if necessary
        dirname = os.path.dirname(self.catalog_outfile)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            logger.info("Created directory: {0}".format(dirname))
        # Save catalog data
        colnames = ["name", "glon", "glat", "ra", "dec",
                    "size_major", "size_minor", "flux",
                    "specindex", "rotation"]
        if os.path.exists(self.catalog_outfile):
            if self.clobber:
                os.remove(self.catalog_outfile)
                logger.warning("Removed existing catalog file: {0}".format(
                    self.catalog_outfile))
            else:
                raise OSError("Output file already exists: {0}".format(
                    self.catalog_outfile))
        self.catalog.to_csv(self.catalog_outfile, columns=colnames,
                            header=True, index=False)
        logger.info("Saved SNRs catalog in use to: %s" % self.catalog_outfile)

    def _filter_catalog(self):
        """
        Filter the catalog data to remove the objects with incomplete data,
        as well as the SNRs lying outside the sky coverage.

        The following cases are filtered out:
        - Missing angular size
        - Missing flux density data
        - Missing spectral index value

        NOTE
        ----
        The objects with uncertain data are currently kept.
        """
        cond1 = pd.isnull(self.catalog["size_major"])
        cond2 = pd.isnull(self.catalog["size_minor"])
        cond3 = pd.isnull(self.catalog["flux"])
        cond4 = pd.isnull(self.catalog["specindex"])
        cond_keep = ~(cond1 | cond2 | cond3 | cond4)
        n_remain = cond_keep.sum()
        n_delete = len(cond_keep) - n_remain
        self.catalog = self.catalog[cond_keep]
        logger.info("SNRs catalog: filtered out due to incomplete data: " +
                    "{0:d} objects".format(n_delete))
        # Filter out the SNRs lying outside the sky region (e.g., a patch)
        skycoords = SkyCoord(l=self.catalog["glon"], b=self.catalog["glat"],
                             frame="galactic", unit="deg")
        inside = self.sky.contains(skycoords)
        n_remain = inside.sum()
        n_delete = len(inside) - n_remain
        self.catalog = self.catalog[inside]
        # Drop the index
        self.catalog.reset_index(drop=True, inplace=True)
        self.catalog_filtered = True
        logger.info("SNRs catalog: filtered out due to sky coverage: " +
                    "{0:d} objects".format(n_delete))
        logger.info("Filtered SNRs catalog: {0} objects".format(n_remain))
        if n_remain == 0:
            raise RuntimeError("NO remaining SNRs within simulation sky! " +
                               "Check the catalog or disable this component.")

    def _add_random_rotation(self):
        """
        Add random rotation angles for each SNR as column "rotation"
        within the catalog data frame.

        The rotation angles are uniformly distributed within [0, 360).

        The rotation happens on the spherical surface, i.e., not with respect
        to the line of sight, but to the Galactic frame coordinate axes.
        """
        num = len(self.catalog)
        angles = np.random.uniform(low=0.0, high=360.0, size=num)
        rotation = pd.Series(data=angles, name="rotation")
        self.catalog["rotation"] = rotation
        logger.info("Added random rotation angles as the 'rotation' column")

    def _calc_Tb(self, flux, specindex, frequency, size):
        """
        Calculate the brightness temperature at requested frequency
        by assuming a power-law spectral shape.

        Parameters
        ----------
        flux : float
            The flux density (unit: [ Jy ]) at the reference
            frequency (i.e., `self.catalog_flux_freq`).
        specindex : float
            The spectral index of the power-law spectrum
        frequency : float
            The frequency (unit: [ MHz ]) where the brightness
            temperature requested.
        size : 2-float tuple
            The (major, minor) axes of the SNR (unit: [ deg ]).
            The order of major and minor can be arbitrary.

        Returns
        -------
        Tb : float
            Brightness temperature at the requested frequency, unit [ K ]

        NOTE
        ----
        The power-law spectral shape is assumed for *flux density* other
        than the *brightness temperature*.
        Therefore, the flux density at the requested frequency should first
        be calculated by extrapolating the spectrum, then convert the flux
        density to derive the brightness temperature.
        """
        freq_ref = self.catalog_flux_freq  # [ MHz ]
        Fnu = flux * (frequency / freq_ref) ** (-specindex)  # [ Jy ]
        omega = size[0] * size[1] * AUC.arcsec2deg**2  # [ arcsec^2 ]
        pixelarea = self.sky.pixelsize ** 2  # [ arcsec^2 ]
        if omega < pixelarea:
            # The object is smaller than a pixel, so round up to a pixel area
            omega = pixelarea
        Tb = Fnu_to_Tb(Fnu, omega, frequency)
        return Tb

    def _simulate_templates(self):
        """
        Simulate the template images/maps for each SNR, and cache these
        templates within this object.

        The template images/maps have values of (or approximate) ones for
        these effective pixels, excluding the pixels corresponding to the
        edges of original rotated ellipse, which may have values of
        significantly less than 1 due to the rotation.

        Therefore, simulating the sky map of one SNR at a requested
        frequency is simply multiplying these cached templates by the
        calculated brightness temperature (Tb) at that frequency.

        Furthermore, the final output sky map of all SNRs are just additions
        of all the maps of each SNR.

        Attributes
        ----------
        templates : dict
            A dictionary containing the simulated templates for each SNR.
            The dictionary keys are the names (`self.catalog["name"]`)
            of the SNRs, and the values are `(idx, val)` tuples with
            `idx` the indexes of effective image pixels and `hpval` the
            values of the corresponding pixels.
            e.g.,
            ``{ name1: (idx1, val1), name2: (idx2, val2), ... }``
        """
        templates = {}
        logger.info("Simulating sky template for each SNR ...")
        for row in self.catalog.itertuples():
            name = row.name
            logger.debug("Simulate sky template for SNR: {0}".format(name))
            gcenter = (row.glon, row.glat)  # [ deg ]
            radii = (int(np.ceil(row.size_major * 0.5 / self.resolution)),
                     int(np.ceil(row.size_minor * 0.5 / self.resolution)))
            rmax = max(radii)
            pcenter = (rmax, rmax)
            image = make_ellipse(pcenter, radii, row.rotation)
            wcs = make_wcs(center=gcenter, size=image.shape,
                           pixelsize=self.resolution,
                           frame="Galactic", projection="CAR")
            idx, val = self.sky.reproject_from(image, wcs, squeeze=True)
            templates[name] = (idx, val)
        logger.info("Done simulate {0} SNR templates".format(len(templates)))
        self.templates = templates

    def _simulate_single(self, data, frequency):
        """
        Simulate one single SNR at the specified frequency.

        Parameters
        ----------
        data : namedtuple
            The data of the SNR to be simulated, given in a ``namedtuple``
            object, from which can get the required properties by
            ``data.key``.
            e.g., elements of `self.catalog.itertuples()`
        frequency : float
            The simulation frequency.
            Unit: [MHz]

        Returns
        -------
        idx : 1D `~numpy.ndarray`
            The indexes of the effective map pixels for the SNR.
        val : 1D `~numpy.ndarray`
            The values (i.e., brightness temperature) of each map
            pixel with respect to the above indexes.

        See Also
        --------
        `self._simulate_template()` for more detailed description.
        """
        name = data.name
        idx, val = self.templates[name]
        # Calculate the brightness temperature
        flux = data.flux
        specindex = data.specindex
        size = (data.size_major/60.0, data.size_minor/60.0)  # [ deg ]
        Tb = self._calc_Tb(flux, specindex, frequency, size)
        val = val * Tb
        return (idx, val)

    def preprocess(self):
        """
        Perform the preparation procedures for the final simulations.

        Attributes
        ----------
        _preprocessed : bool
            This attribute presents and is ``True`` after the preparation
            procedures are performed.
        """
        if hasattr(self, "_preprocessed") and self._preprocessed:
            return

        logger.info("{name}: preprocessing ...".format(name=self.name))
        self._load_catalog()
        self._filter_catalog()
        self._add_random_rotation()
        # Simulate the template maps for each SNR
        self._simulate_templates()

        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """
        Simulate the sky map of all Galactic SNRs emission at the specified
        frequency.

        Parameters
        ----------
        frequency : float
            The simulation frequency.
            Unit: [MHz]

        Returns
        -------
        sky : `~SkyBase`
            The simulated sky image as a new sky instance.

        See Also
        --------
        `self._simulate_template()` for more detailed description.
        """
        logger.info("Simulating {name} map at {freq:.2f} [MHz] ...".format(
            name=self.name, freq=frequency))
        sky = self.sky.copy()
        sky.frequency = frequency
        for row in self.catalog.itertuples():
            index, value = self._simulate_single(row, frequency)
            sky.data[index] += value
        logger.info("Done simulate map at %.2f [MHz]." % frequency)
        return sky

    def simulate(self, frequencies=None):
        """
        Simulate the sky maps of all Galactic SNRs emission.

        Parameters
        ----------
        frequencies : float, or list[float]
            The frequencies where to simulate the foreground map.
            Unit: [MHz]
            Default: None (i.e., use ``self.frequencies``)

        Returns
        -------
        skyfiles : list[str]
            List of the filepath to the written sky files
        """
        if frequencies is None:
            frequencies = self.frequencies
        else:
            frequencies = np.array(frequencies, ndmin=1)

        logger.info("Simulating {name} ...".format(name=self.name))
        skyfiles = []
        for freq in frequencies:
            sky = self.simulate_frequency(freq)
            outfile = self._outfilepath(frequency=freq)
            sky.write(outfile)
            skyfiles.append(outfile)
        logger.info("Done simulate {name}!".format(name=self.name))
        return skyfiles

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        logger.info("{name}: postprocessing ...".format(name=self.name))
        # Save the catalog actually used in the simulation
        self._save_catalog_inuse()
