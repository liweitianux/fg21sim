=============
Template Data
=============

The simulation of the following foreground components requires specific
template map(s) and/or observational/simulation catalog(s) as the input:

* ``galactic/synchrotron``:
  requires the Haslam 408 MHz survey as the template map, and the
  spectral index map.
* ``galactic/freefree``:
  requires the Hα map and the dust map.
* ``galactic/snr``:
  requires the catalog of the Galactic SNRs.

The following template data are provided:

* Haslam 408 MHz all-sky survey:
  `haslam408_dsds_Remazeilles2014_ns512.fits.xz <https://github.com/liweitianux/fg21sim/raw/master/data/haslam408_dsds_Remazeilles2014_ns512.fits.xz>`_,
  8079712 bytes,
  (MD5) ``da895a58a19701545745d0e75d91a098``

* Galactic synchrotron spectral index all-sky map:
  `synchrotron_specind2_ns512.fits.xz <https://github.com/liweitianux/fg21sim/raw/master/data/synchrotron_specind2_ns512.fits.xz>`_,
  7180244 bytes,
  (MD5) ``1ae30d91facb8537ccc9a7f17c065f0a``

* Galactic Hα all-sky map:
  `Halpha_fwhm06_ns1024.fits.xz <https://github.com/liweitianux/fg21sim/raw/master/data/Halpha_fwhm06_ns1024.fits.xz>`_,
  36881116 bytes,
  (MD5) ``7ab5df0623728d2ad67281ec7b95c5a0``

* Galactic dust all-sky map:
  `SFD_i100_ns1024.fits.xz <https://github.com/liweitianux/fg21sim/raw/master/data/SFD_i100_ns1024.fits.xz>`_,
  36734292 bytes,
  (MD5) ``da7409e0b9215e0bf4b39a7c0c079e08``

* Galactic SNR catalog:
  `GalacticSNRs_Green2014.csv <https://github.com/liweitianux/fg21sim/raw/master/data/GalacticSNRs_Green2014.csv>`_,
  24981 bytes,
  (MD5) ``9851be15145e77f7f49f301ee54e5a14``

The above all-sky maps are in HEALPix format.
Sky patches can be cut out from them using the ``get-healpix-patch`` tool.
The ``fg21sim-download-data`` tool can also be used to retrieve the above
template data by providing with the `data manifest <../data/manifest.json>`_.
