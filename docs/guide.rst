==========
User Guide
==========

This is a simple guide on how to use the **fg21sim** package to carry
out the foregrounds simulation, which produces the all-sky maps of the
enabled foreground components.

The simulation of several foreground components uses the semi-empirical
method, thus it requires specific template map(s) and/or
observational/simulation catalog(s) as the input:

* ``galactic/synchrotron``:
  requires the Haslam 408 MHz survey as the template map, and the
  spectral index map.
* ``galactic/freefree``:
  requires the Hα map and the dust map.
* ``galactic/snr``:
  requires the catalog of the Galactic SNRs.
* ``extragalactic/clusters``:
  requires the catalog of the clusters of galaxies.

All the required input templates and catalogs can be retrieved using
the ``fg21sim-download-data`` CLI tool, by providing it with this
`data manifest <data-manifest.json>`_.

Then, a configuration file is required to run the foregrounds simulation,
which controls all aspects of the simulation behaviors.
There are two types of configuration options:
*required* (which require the user to explicitly provide the values)
and *optional* (which already have sensible defaults, however, the user
can also override them).

There is an `example configuration file <fg21sim.conf.example>`_ with
detailed explanations on each configuration option.
Also there is a brief `test configuration file <fg21sim-test.conf>`_
which may be useful to test whether this package is correctly installed
and runs smoothly.

Finally, the foregrounds simulation can be kicked off using the CLI tool::

    $ fg21sim --logfile fg21sim.log fg21sim.conf

This way, the simulation program will take configurations from
file ``fg21sim.conf``, and log messages to both the screen and file
``fg21sim.log``.


On the other hand, the Web UI can also be used, which provides a more
intuitive and friendly way to tune the configurations, to view the
logging messages, as well as to navigate the simulation products::

    $ fg21sim-webui --debug &

Then the Web UI can be accessed at ``http://localhost:21127``.
