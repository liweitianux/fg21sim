==========
User Guide
==========

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

