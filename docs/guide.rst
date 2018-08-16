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
Please refer to the `configuration specification file <fg21sim.conf.spec>`_
for more information on the available options.
Also there is a brief `test configuration file <fg21sim-test.conf>`_
which may be useful to test whether this package is correctly installed
and runs smoothly.

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
`Wang et al. 2010`_.


----------------------
Observation Simulation
----------------------

In order to incorporate the instrumental effects into the simulated
sky maps, the latest `SKA1-Low layout configuration`_ (released on
2016 May 21) is employed to carry out the observation simulation.

The `OSKAR`_ simulator is used to perform the interferometric
observations.  The ``make-ska1low-model`` tool was writen to generate
the *sky model* of the SKA1-Low for use by ``OSKAR``.
The simulated *visibility data* are then imaged by utilizing the
`WSClean`_ to generate the "observed" images.

The scripts that help carry out the observation simulations can be
found at `atoolbox/astro/oskar`_.


.. _`pointsource tools`:
   https://github.com/liweitianux/radio-fg-simu-tools/tree/master/pointsource
.. _`Wang et al. 2010`_:
   http://adsabs.harvard.edu/abs/2010ApJ...723..620W
.. _`SKA1-Low layout configuration`:
   https://astronomers.skatelescope.org/wp-content/uploads/2016/09/SKA-TEL-SKO-0000422_02_SKA1_LowConfigurationCoordinates-1.pdf
.. _`OSKAR`_:
   https://github.com/OxfordSKA/OSKAR
.. _`WSClean`_:
   https://sourceforge.net/projects/wsclean/
.. _`atoolbox/astro/oskar`:
   https://github.com/liweitianux/atoolbox/tree/master/astro/oskar
