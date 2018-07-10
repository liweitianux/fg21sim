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
  `haslam408_dsds_Remazeilles2014.fits <https://onedrive.live.com/download?cid=6BC61834227AC6CE&resid=6BC61834227AC6CE%2116032&authkey=AHJyQFnMlp8yOIo>`_,
  12594240 bytes,
  (MD5) 53b99b6e61b80f6c6a603c0a50c9ba51

* Galactic synchrotron spectral index all-sky map:
  `synchrotron_specind2.fits <https://onedrive.live.com/download?cid=6BC61834227AC6CE&resid=6BC61834227AC6CE%2116033&authkey=ADmJvJ3Shy7M9ig>`_,
  25176960 bytes,
  (MD5) 0bc9899915db805ec1709f0c83ca6617

* Galactic Hα all-sky map:
  `Halpha_fwhm06_1024.fits <https://onedrive.live.com/download?cid=6BC61834227AC6CE&resid=6BC61834227AC6CE%2116034&authkey=AGjsiwZBaZ-ZZLE>`_,
  50342400 bytes,
  (MD5) 1a3ec062818bbdd8254e5158cce90652

* Galactic dust all-sky map:
  `SFD_i100_ns1024.fits <https://onedrive.live.com/download?cid=6BC61834227AC6CE&resid=6BC61834227AC6CE%2116029&authkey=AAN7DT0JKWpFlyA>`_,
  50342400 bytes,
  (MD5) e9d6e683f9f6aaa308275196615db17d

* Galactic SNR catalog:
  `GalacticSNRs_Green2014.csv <https://onedrive.live.com/download?cid=6BC61834227AC6CE&resid=6BC61834227AC6CE%2116026&authkey=AJcYjHaI7O7FEcY>`_,
  24981 bytes,
  (MD5) 9851be15145e77f7f49f301ee54e5a14

The above all-sky maps are in HEALPix format.
Sky patches can be cut out from them using the ``get-healpix-patch`` tool.
The ``fg21sim-download-data`` tool can also be used to retrieve the above
template data by providing with the `data manifest <data-manifest.json>`_.
