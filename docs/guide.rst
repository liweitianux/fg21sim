==========
User Guide
==========

-----------
Get Started
-----------

This is a simple guide on how to use the **fg21sim** package to carry
out the foregrounds simulation, which produces the all-sky maps or sky
patches of the enabled foreground components.

The simulation of the following foreground components requires specific
template map(s) and/or observational/simulation catalog(s) as the input:

* ``galactic/synchrotron``:
  requires the Haslam 408 MHz survey as the template map, and the
  spectral index map.
* ``galactic/freefree``:
  requires the Hα map and the dust map.
* ``galactic/snr``:
  requires the catalog of the Galactic SNRs.

See the `template data <data.rst>`_ page for more details on the input
data and how to obtain them.

Then, a configuration file is required to run the foregrounds simulation,
which controls all aspects of the simulation behaviors.
There are two types of configuration options:
*required* (which require the user to explicitly provide the values)
and *optional* (which already have sensible defaults, however, the user
can also override them).
Please refer to the
`configuration specification file <../fg21sim/configs/config.spec>`_
for more information on the available options.

Finally, the foregrounds simulation can be kicked off by executing::

    $ fg21sim --logfile fg21sim.log fg21sim.conf

The program will read configurations from file ``fg21sim.conf``, and log
messages to both the screen and file ``fg21sim.log``.


---------
Sky Patch
---------

When doing high-resolution simulations, it is more appropriate to specify
a sky patch (e.g., 10x10 deg^2) instead of using the whole sky.

For simulating Galactic diffuse emission maps, the template sky patches
are required as the input.  To generate such sky patches, use the
``get-healpix-patch`` tool to extract the needed patch from the all-sky
HEALPix template maps.


-------------
Point Sources
-------------

The sky maps of extragalactic point sources can be simulated using
these `pointsource tools`_ that were developed for use in
`Wang et al. 2010`_ .


----------------------
Observation Simulation
----------------------

In order to incorporate the instrumental effects into the simulated
sky maps, the latest `SKA1-Low layout configuration`_ (released on
2016 May 21) is employed to carry out the observation simulation.

The `OSKAR`_ simulator is used to perform the interferometric
observations.  The ``make-ska1low-model`` tool was writen to generate
the *telescope model* of the SKA1-Low for use by ``OSKAR``.
The simulated *visibility data* are then imaged by utilizing the
`WSClean`_ to generate the "observed" images.

1. Generate the *telescope model* for the OSKAR simulator::

    $ mkdir telescopes
    $ make-ska1low-model -o telescopes/ska1low.tm

2. Convert the simulated sky map (``example.fits``) into *sky model*
   for OSKAR::

    $ fits2skymodel.py example.fits

   The ``fits2skymodel.py`` tool will obtain the pixel size and
   frequency from the input FITS image header.
   This produces the OSKAR sky model file named as ``example.osm``.

3. Perform the observation simulation::

    $ run_oskar.py -c sim_interferometer.base.ini -l <freq>:example.osm

   The ``<freq>`` is the frequency (in units of MHz) of the input
   sky image.
   The ``sim_interferometer.base.ini`` is the basic configuration
   file for ``oskar_sim_interferometer`` tool in OSKAR.
   Here is an `example config file <https://github.com/liweitianux/atoolbox/blob/master/astro/oskar/sim_interferometer.base.ini>`_.
   The simulated visibility data will be located at
   ``visibility/example.ms``.

4. Create image from the visibility data::

    $ wsclean.py --niter 1000000 --weight briggs \
          --size <npix> --pixelsize <pixelsize/arcsec> \
          --taper-gaus <2*pixelsize> --circular-beam \
          --threshold-nsigma 2.5 \
          --name example \
          --ms visibility/example.ms

   The created clean image is thus ``example-image.fits``.
   The created images have unit of ``Jy/beam`` and can be converted
   to have unit of ``K`` by using::

    $ jybeam2k.py example-image.fits example-imageK.fits

The above used tools that help carry out the observation
simulations can be found at `atoolbox/astro/oskar`_.

**NOTE**:
A sky image cube including multiple frequency channels must be
simulated one frequency at a time.


-------------
Data Analysis
-------------

Images of a set of frequency channels can be combined to create
an image cube by using::

    $ fitscube.py create -z <start-freq> -s <channel-width> -u Hz \
          -o example-cube.fits -i *-imageK.fits

The power spectrum of the image cube can then be calculated::

    $ ps2d.py -i example-cube.fits -o example-ps2d.fits
    $ ps1d_eorwindow.py [options] -i example-ps2d.fits -o example-ps1d.txt

There are other scripts that can help analyze the results, such as
``fitsimage.py``, ``eor_window.py``, ``calc_psd.py``.

All the above mentioned tools can be found at `atoolbox/astro`_
and the sub-directories there.


.. _pointsource tools:
   https://github.com/liweitianux/radio-fg-simu-tools/tree/master/pointsource
.. _Wang et al. 2010:
   http://adsabs.harvard.edu/abs/2010ApJ...723..620W
.. _SKA1-Low layout configuration:
   https://astronomers.skatelescope.org/wp-content/uploads/2016/09/SKA-TEL-SKO-0000422_02_SKA1_LowConfigurationCoordinates-1.pdf
.. _OSKAR:
   https://github.com/OxfordSKA/OSKAR
.. _WSClean:
   https://sourceforge.net/projects/wsclean/
.. _atoolbox/astro/oskar:
   https://github.com/liweitianux/atoolbox/tree/master/astro/oskar
.. _atoolbox/astro:
   https://github.com/liweitianux/atoolbox/tree/master/astro
