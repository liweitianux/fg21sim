# Copyright (c) 2016-2017,2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Manage and manipulate the simulation products.
"""

import os
import shutil
import json
import logging
from collections import OrderedDict

import numpy as np

from .errors import ManifestError
from .utils.hashutil import calc_md5


logger = logging.getLogger(__name__)


class Products:
    """
    Manage and manipulate the simulation products.

    Parameters
    ----------
    manifestfile : str, optional
        The absolute path to the manifest file for loading.
    load : bool, optional
        Load the specified manifest file if ``True``.

    Attributes
    ----------
    manifest : dict
        The manifest of the simulation products.
        See the below "Manifest Format" section for more details.
    manifestfile : str
        The absolute path to the loaded manifest file.

    Manifest Format
    ---------------
    ``
    {
        "frequency" : {
            "frequencies" : [ <list of frequencies> ],
            "id" : [ <id/index of each frequency> ],
            "unit": <frequency unit>,
        },
        <component> : [
            {
                "frequency" : <frequency>,
                "healpix" : {
                    "path" : <relative path to healpix file>,
                    "size" : <file size (bytes)>,
                    "md5" : <md5 checksum>,
                },
                "hpx" : {
                    "path" : <relative path to converted HPX image>,
                    "size" : <file size (bytes)>,
                    "md5" : <md5 checksum>,
                },
            },
            ...
        ],
        ...
    }
    ``
    """
    def __init__(self, manifestfile=None, load=True):
        self.manifest = OrderedDict()
        self.manifestfile = manifestfile
        if (manifestfile is not None) and load:
            self.load(manifestfile)

    @property
    def frequencies(self):
        """
        Get the frequencies of the products from the manifest.
        """
        return self.manifest["frequency"]

    @frequencies.setter
    def frequencies(self, value):
        """
        Set the frequencies of the products and store in the manifest.

        Each frequency has an ID (also its index in the frequencies list).

        Parameters
        ----------
        value : list[float], or tuple(list[float], str)
            The list of simulation frequencies, or a tuple of the frequencies
            and its unit (default: MHz).
        """
        if isinstance(value, tuple) and len(value) == 2:
            frequencies, unit = value
        else:
            frequencies = value
            unit = "MHz"
        #
        self.manifest["frequency"] = {
            "frequencies": list(frequencies),
            "id": list(range(len(frequencies))),
            "unit": unit,
        }
        logger.info("Number of frequencies: {0}, ".format(len(frequencies)) +
                    "unit: {0}".format(unit))

    def find_frequency_id(self, frequency, atol=1e-3):
        """
        Find the ID of the given frequency by comparing it to the
        frequencies list.

        Parameters
        ----------
        frequency : float
            The input frequency for which to find its the ID
        atol : float, optional
            The absolute tolerance parameter.
            For finite values, isclose uses the following equation to test
            whether two floating point values are equivalent:
                absolute(a - b) <= (atol + rtol * absolute(b))

        Returns
        -------
        id : int
            The ID of the input frequency, and ``-1`` if not found.
            ``None`` if frequencies currently not set.
        """
        try:
            frequencies = np.asarray(self.frequencies["frequencies"])
            fid = self.frequencies["id"]
            res = np.where(np.isclose(frequencies, frequency, atol=atol))[0]
            if len(res) == 0:
                return -1
            else:
                return fid[res[0]]
        except KeyError:
            # Frequencies currently not set
            return None

    def add_product(self, comp_id, freq_id, filepath):
        """
        Add one single simulation product to the manifest.

        The metadata (file path, size and MD5 checksum) of simulation
        products are stored in the manifest.

        Parameters
        ----------
        comp_id : str
            ID of the component to be added.
        freq_id : int
            Frequency ID
        filepath : str
            File path of the product (HEALPix maps).

        Raises
        ------
        ManifestError :
            The attribute ``self.manifestfile`` is not set.
        """
        if self.manifestfile is None:
            raise ManifestError("'self.manifestfile' is not set")

        frequencies = self.frequencies["frequencies"]
        if comp_id not in self.manifest.keys():
            self.manifest[comp_id] = [{} for i in range(len(frequencies))]

        root_dir = self.get_root_dir()
        self.manifest[comp_id][freq_id] = {
            "frequency": frequencies[freq_id],
            "healpix": {
                # Relative path to the HEALPix map file from this manifest
                "path": os.path.relpath(filepath, root_dir),
                "size": os.path.getsize(filepath),  # [byte]
                "md5": calc_md5(filepath),
            }
        }
        logger.info("Added one product to the manifest: {0}".format(filepath))

    def add_component(self, comp_id, paths):
        """
        Add a simulation component to the manifest.

        The simulation products (file path, size and MD5 checksum) are
        stored in the manifest.

        Parameters
        ----------
        comp_id : str
            ID of the component to be added.
        paths : list[str]
            List of the file paths of the component products (HEALPix maps).

        Raises
        ------
        ManifestError :
            * The attribute ``self.manifestfile`` is not set.
            * Number of input paths dose NOT equal to number of frequencies
        """
        if self.manifestfile is None:
            raise ManifestError("'self.manifestfile' is not set")

        frequencies = self.frequencies["frequencies"]
        if len(paths) != len(frequencies):
            raise ManifestError("Number of paths (%d) != " % len(paths) +
                                "number of frequencies")

        for freq_id, filepath in enumerate(paths):
            self.add_product(comp_id, freq_id, filepath)
        logger.info("Added component '{0}' to the manifest".format(comp_id))

    def checksum(self, comp_id, freq_id):
        """
        Calculate the checksum for products and compare with the existing
        manifest.

        Parameters
        ----------
        comp_id : str
            ID of the component whose product will be checksum'ed
        freq_id : int
            The frequency ID of the specific product within the component.

        Returns
        -------
        match : bool
            Whether the MD5 checksum of the on-disk product matches the
            checksum stored in the manifest.
        hash : str
            The MD5 checksum value of the on-disk product.
        """
        root_dir = self.get_root_dir()
        metadata = self.get_product(comp_id, freq_id)
        filepath = os.path.join(root_dir, metadata["healpix"]["path"])
        hash_true = metadata["healpix"]["md5"]
        hash_ondisk = calc_md5(filepath)
        return (hash_ondisk == hash_true, hash_ondisk)

    def get_root_dir(self):
        """
        Get the root directory where the products locate, which is also
        the directory where the manifest file locates.

        Returns
        -------
        root_dir : str
            The absolute path of the products root directory.

        Raises
        ------
        ManifestError :
            The manifest currently not loaded, thus unable to determine
            the products root directory.
        """
        try:
            return os.path.dirname(self.manifestfile)
        except AttributeError:
            raise ManifestError("Manifest currently not loaded!")

    def get_product(self, comp_id, freq_id):
        return self.manifest[comp_id][freq_id]

    def get_product_abspath(self, comp_id, freq_id, ptype="healpix"):
        """
        Get the absolute path to the specified product.

        Parameters
        ----------
        comp_id : str
            ID of the component whose product will be checksum'ed
        freq_id : int
            The frequency ID of the specific product within the component.
        ptype : str, optional
            The type of product whose path will be retrieved.
            Valid values: ``healpix`` (default), ``hpx``.

        Returns
        -------
        abspath : str
            The absolute path to the specified product

        Raises
        ------
        ValueError :
            Invalid ``ptype`` other than ``healpix`` and ``hpx``.
        KeyError :
            The requested product type not available (e.g., the HPX
            image is not generated yet)
        """
        if ptype not in ["healpix", "hpx"]:
            raise ValueError("Invalid ptype: {0}".format(ptype))

        root_dir = self.get_root_dir()
        metadata = self.get_product(comp_id, freq_id)
        abspath = os.path.join(root_dir, metadata[ptype]["path"])
        return abspath

    def convert_hpx(self, comp_id, freq_id, clobber=False):
        """
        Convert the specified HEALPix map product to HPX projected FITS image.
        Also add the metadata of the HPX image to the manifest.

        Raises
        ------
        IOError :
            Output HPX image already exists and ``clobber=False``
        """
        from astropy.io import fits
        from .utils.healpix import healpix2hpx

        root_dir = self.get_root_dir()
        metadata = self.get_product(comp_id, freq_id)
        infile = os.path.join(root_dir, metadata["healpix"]["path"])
        outfile = os.path.splitext(infile)[0] + "_hpx.fits"
        if os.path.exists(outfile):
            if clobber:
                os.remove(outfile)
                logger.warning("Removed existing HPX image: %s" % outfile)
            else:
                raise IOError("Output HPX image already exists: %s" % outfile)

        # Convert HEALPix map to HPX projected FITS image
        logger.info("Converting HEALPix map to HPX image: %s" % infile)
        hpx_data, hpx_header = healpix2hpx(infile)
        hdu = fits.PrimaryHDU(data=hpx_data, header=hpx_header)
        hdu.writeto(outfile)
        logger.info("Converted HEALPix map to HPX image: %s" % outfile)

        size = os.path.getsize(outfile)
        md5 = calc_md5(outfile)
        metadata["hpx"] = {
            "path": os.path.relpath(outfile, root_dir),
            "size": size,
            "md5": md5,
        }
        return (outfile, size, md5)

    def reset(self):
        self.manifest = OrderedDict()
        self.manifestfile = None
        logger.warning("Reset products manifest!")

    def dump(self, outfile=None, clobber=False, backup=True):
        """
        Dump the manifest as a JSON file.

        Parameters
        ----------
        outfile : str, optional
            The path to the output manifest file.
            If not provided, then use ``self.manifestfile``.
            NOTE:
            This must be an *absolute path*.
            Prefix ``~`` (tilde) is allowed and will be expanded.
        clobber : bool, optional
            Overwrite the output file if already exists.
        backup : bool, optional
            Backup the output file with suffix ``.old`` if already exists.

        Raises
        ------
        ValueError :
            The given ``outfile`` is NOT an absolute path.
            Or the ``self.manifestfile`` is ``None`` while the ``outfile``
            is missing.
        OSError :
            If the target filename already exists and ``clobber=False``.
        """
        if outfile is None:
            if self.manifestfile is None:
                raise ValueError("outfile is missing and " +
                                 "self.manifestfile is None")
            else:
                outfile = self.manifestfile
                logger.info("Output to self.manifestfile: {0}".format(outfile))
        outfile = os.path.expanduser(outfile)
        if not os.path.isabs(outfile):
            raise ValueError("Not an absolute path: {0}".format(outfile))
        if os.path.exists(outfile):
            if clobber:
                # Make a backup with suffix ``.old``
                backfile = outfile + ".old"
                shutil.copyfile(outfile, backfile)
                logger.info("Backed up old manifest file as: " + backfile)
            else:
                raise OSError("File already exists: {0}".format(outfile))

        with open(outfile, "w") as fp:
            json.dump(self.manifest, fp, indent=4)
            fp.write("\n")
        logger.info("Dumped manifest to file: {0}".format(outfile))

    def load(self, infile):
        """
        Load the manifest from a JSON file.

        Parameters
        ----------
        infile : str
            The path to the input manifest file.
            NOTE:
            This must be an *absolute path*.
            Prefix ``~`` (tilde) is allowed and will be expanded.

        Raises
        ------
        ValueError :
            The given ``infile`` is NOT an absolute path.
        OSError :
            Cannot read the input manifest file.
        """
        infile = os.path.expanduser(infile)
        if not os.path.isabs(infile):
            raise ValueError("Not an absolute path: {0}".format(infile))

        self.reset()
        self.manifest = json.load(open(infile), object_pairs_hook=OrderedDict)
        self.manifestfile = infile
        logger.info("Loaded manifest from file: {0}".format(infile))
