# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Radio interferometer layout configurations.
"""

import os
import logging
import shutil

import numpy as np
import pandas as pd

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

from .wgs84 import geodetic2enu


logger = logging.getLogger(__name__)


class SKA1Low:
    """
    Process SKA1-low layout data and generate the telescope model
    for OSKAR.

    Parameters
    ----------
    infile : str
        Path to the SKA1-low layout data file
    stn_antennas : int, optional
        Number of antenna elements per station (default: 256)
    stn_diameter : float, optional
        Diameter of each station (unit: [m])
    ant_min_sep : float, optional
        Minimum separation between two antennas (unit: [m])
    r_core : float, optional
        Radius defined as the core region (unit: [m]), default: 500.0
    r_central : float, optional
        Radius defined as the central region (unit: [m]), default: 1700.0

    Reference
    ---------
    [1] SKA-TEL-SKO-0000422, revision 02, 2016-05-31, Table 1
        http://astronomers.skatelescope.org/wp-content/uploads/2016/09/SKA-TEL-SKO-0000422_02_SKA1_LowConfigurationCoordinates-1.pdf
    [2] OSKAR: telescope model
        http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Telescope-Model.pdf
    """
    def __init__(self, infile, stn_antennas=256, stn_diameter=45.0,
                 ant_min_sep=1.5, r_core=500.0, r_central=1700.0):
        self.infile = infile
        self.stn_antennas = stn_antennas
        self.stn_diameter = stn_diameter  # [m]
        self.ant_min_sep = ant_min_sep  # [m]
        self.r_core = r_core  # [m]
        self.r_central = r_central  # [m]
        self.data = pd.read_csv(infile, sep="\s+", comment="#",
                                index_col="Label")
        logger.info("Read telescope layout data from: %s" % infile)
        self.position_wgs84 = np.array(self.data.loc["CENTER", :])
        logger.info("Telescope center coordinate: (%f, %f)" %
                    tuple(self.position_wgs84))
        self.labels = self.make_station_labels(self.data.index[1:])
        # (longitudes, latitudes)
        self.layouts_wgs84 = np.array(self.data.iloc[1:, :])
        # Convert WGS84 to ENU coordinates
        p0 = [self.position_wgs84[0], self.position_wgs84[1], 0.0]
        layouts = np.array([geodetic2enu((lon, lat, 0.0), p0)
                            for lon, lat in self.layouts_wgs84])
        layouts[:, 2] = 0.0  # set `up` to 0.0
        self.layouts_enu = layouts
        logger.info("Number of stations: %d" % len(self.layouts_wgs84))

    def generate_stations(self):
        """
        Generate the antenna elements layouts for each station.
        """
        layouts = []
        N = len(self.labels)
        logger.info("Number of antennas per station: %d" %
                    self.stn_antennas)
        logger.info("Station diameter: %.2f [m]" % self.stn_diameter)
        logger.info("Station antennas minimum separation: %.2f [m]" %
                    self.ant_min_sep)
        logger.info("Generating antenna elements layouts ...")
        for i, label in enumerate(self.labels):
            logger.debug("Generate layout for [#%d/%d] station: %s" %
                         (i+1, N, label))
            x, y, __ = self.rand_uniform_2d(
                n=self.stn_antennas, r_max=self.stn_diameter/2.0,
                min_sep=self.ant_min_sep)
            layouts.append((x, y))
        self.stn_layouts = layouts
        logger.info("DONE generate station layouts.")

    def plot_stations(self, outdir, figsize=(8, 8), dpi=150):
        """
        Make a plot for each station.
        """
        if not has_matplotlib:
            logger.error("matplotlib required to plot stations")

        N = len(self.labels)
        r_max = self.stn_diameter / 2.0
        for i, label in enumerate(self.labels):
            x, y = self.stn_layouts[i]
            fpng = os.path.join(outdir, label+".png")
            fig = Figure(figsize=figsize, dpi=dpi)
            FigureCanvas(fig)
            ax = fig.add_subplot(111, aspect="equal")
            ax.plot(x, y, "k+")
            ax.grid()
            ax.set_xlim((-r_max*1.05, r_max*1.05))
            ax.set_ylim((-r_max*1.05, r_max*1.05))
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            ax.set_title("Antenna elements: %d; $d_{min}$ = %.2f [m]" %
                         (self.stn_antennas, self.ant_min_sep),
                         fontsize=10)
            fig.suptitle("Station [#%d/%d]: %s" % (i+1, N, label),
                         fontsize=14)
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            fig.savefig(fpng)
            logger.debug("Made plot for [#%d/%d] station: %s" %
                         (i+1, N, fpng))

    def plot_telescope(self, outdir, figsize=(8, 8), dpi=150):
        """
        Make plots showing all the telescope stations, central
        stations, and core stations.
        """
        if not has_matplotlib:
            logger.error("matplotlib required to plot the telescope")

        x, y = self.layouts_enu[:, 0], self.layouts_enu[:, 1]
        # All stations
        fpng = os.path.join(outdir, "layout_all.png")
        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvas(fig)
        ax = fig.add_subplot(111, aspect="equal")
        ax.plot(x, y, "ko")
        ax.grid()
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("SKA1-low Stations Layout (All #%d)" % len(x))
        fig.tight_layout()
        fig.savefig(fpng)
        logger.debug("Made plot for telescope all station: %s" % fpng)
        # TODO...

    def make_oskar_model(self, outdir, clobber=False):
        """
        Create the telescope model for OSKAR.
        """
        if os.path.exists(outdir):
            if clobber:
                shutil.rmtree(outdir)
                logger.warning("Removed existing model: %s" % outdir)
            else:
                raise FileExistsError("Output directory already exists: " %
                                      outdir)
        os.mkdir(outdir)
        logger.info("Created telescope model at: %s" % outdir)
        # Write position
        fposition = os.path.join(outdir, "position.txt")
        open(fposition, "w").writelines([
            "# SKA1-low layout: %s\n" % self.infile,
            "# Telescope center position (WGS84)\n",
            "# longitude[deg]  latitude[deg]\n",
            "%.8f  %.8f\n" % tuple(self.position_wgs84)
        ])
        logger.info("Wrote telescope position: %s" % fposition)
        # Write layout of stations
        flayout = os.path.join(outdir, "layout.txt")
        header = ["SKA1-low layout: %s" % self.infile,
                  "All stations layout",
                  "East[m]   North[m]   Up[m]"]
        np.savetxt(flayout, self.layouts_enu, header="\n".join(header))
        logger.info("Wrote station layouts: %s" % flayout)
        # Write stations
        N = len(self.labels)
        for i, label in enumerate(self.labels):
            stn_dir = os.path.join(outdir, label)
            os.mkdir(stn_dir)
            fstation = os.path.join(stn_dir, "layout.txt")
            header = [
                "Antenna elements layout",
                "Station label: %s" % label,
                "Number of antennas: %d" % self.stn_antennas,
                "Station diameter: %.2f [m]" % self.stn_diameter,
                "Antenna minimum separation: %.2f [m]" % self.ant_min_sep,
                "X[m]  Y[m]"
            ]
            np.savetxt(fstation, np.column_stack(self.stn_layouts[i]),
                       header="\n".join(header))
            logger.debug("Wrote layout for [#%d/%d] station: %s" %
                         (i+1, N, fstation))
        logger.info("DONE wrote telescope model: %s" % outdir)

    @staticmethod
    def make_station_labels(labels, base="stn"):
        """
        Make the labels for each station, which will also be used
        as the sub-directory names for the output telescope model.
        """
        N = len(labels)
        ndigits = int(np.log10(N)) + 1
        fmt = "{base}.%(id)0{ndigits}d.%(label)s".format(
            base=base, ndigits=ndigits)
        stnlabels = [fmt % {"id": i+1, "label": l}
                     for i, l in enumerate(labels)]
        return stnlabels

    @staticmethod
    def rand_uniform_2d(n, r_max, min_sep, r_min=None):
        """
        Generate 2D random points with a minimum separation within
        a radius range (i.e., annulus/circle).

        Credit:
        * https://github.com/OxfordSKA/SKA1-low-layouts :
          layouts/utilities/layout.py - Layout.rand_uniform_2d()
        """
        grid_size = min(100, int(np.ceil(r_max * 2.0) / min_sep))
        grid_cell = r_max * 2.0 / grid_size
        scale = 1.0 / grid_cell

        x, y = np.zeros(n), np.zeros(n)
        grid = {
            "start": np.zeros((grid_size, grid_size), dtype=np.int),
            "end": np.zeros((grid_size, grid_size), dtype=np.int),
            "count": np.zeros((grid_size, grid_size), dtype=np.int),
            "next": np.zeros(n, dtype=int)
        }

        num_tries, max_tries, total_tries = 0, 0, 0
        for j in range(n):
            done = False
            while not done:
                xt, yt = np.random.rand(2) * 2*r_max - r_max
                rt = (xt**2 + yt**2) ** 0.5
                if rt + (min_sep/2.0) > r_max:
                    num_tries += 1
                elif r_min and rt - (min_sep/2.0) < r_min:
                    num_tries += 1
                else:
                    jx = int(round(xt + r_max) * scale)
                    jy = int(round(yt + r_max) * scale)
                    x0 = max(0, jx-2)
                    x1 = min(grid_size, jx+3)
                    y0 = max(0, jy-2)
                    y1 = min(grid_size, jy+3)
                    # Find the minimum spacing between the trial point
                    # and other points
                    d_min = r_max * 2.0
                    for ky in range(y0, y1):
                        for kx in range(x0, x1):
                            if grid["count"][ky, kx] > 0:
                                i_other = grid["start"][ky, kx]
                                for kh in range(grid["count"][ky, kx]):
                                    dx = xt - x[i_other]
                                    dy = yt - y[i_other]
                                    d_other = (dx**2 + dy**2) ** 0.5
                                    d_min = min(d_min, d_other)
                                    i_other = grid["next"][i_other]
                    if d_min >= min_sep:
                        x[j], y[j] = xt, yt
                        if grid["count"][jy, jx] == 0:
                            grid["start"][jy, jx] = j
                        else:
                            grid["next"][grid["end"][jy, jx]] = j
                        grid["end"][jy, jx] = j
                        grid["count"][jy, jx] += 1
                        max_tries = max(max_tries, num_tries)
                        total_tries += num_tries
                        num_tries = 0
                        done = True
                    else:
                        num_tries += 1

        info = {"max_tries": max_tries, "total_tries": total_tries}
        return (x, y, info)
