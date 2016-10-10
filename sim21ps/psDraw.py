#!/usr/bin/env python3
#
# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
This module namely psDraw is designed to read csv format ps lists,
calculate the flux and surface brightness of the sources at different
frequency, and then display them on the image mat.

Modules
-------
Basic modules: numpy, pandas, PIL
Custom designed modules: basic_parameters, psCatelogue

Classes
-------
Flux: class
    A class to calculate the ps's surface brightness accordingly
    to its frequency and distance.

Functions
---------
read_csv: read the csv format files, judge the ps type and
transformed to be iterable numpy.ndarray.

calc_flux: calculate the flux and surface brightness of the ps.

draw_elp: processing on the elliptical and circular core or lobes.

draw_ps: draw the ps on the image map.
"""

# Modules
import sys
import getopt
import numpy as np
import astropy.units as au
import healpy as hp
# from pandas import DataFrame
import pandas as pd
# Cumstom designed modules
from fg21sim.utils import write_fits_healpix
# import basic_params
# import psCatelogue

# Init
# Params = basic_params.PixelParams(img_size)


class Flux:
    """
    To calculate the flux and surface brightness of the point sources
    accordingly to its type and frequency

    Parameters
    ----------
    Freq: float
        The frequency
    ClassType: int
        The type of point source, which is default as 1.
        | ClassType | Code |
        |:---------:|:----:|
        |    SF     |  1   |
        |    SB     |  2   |
        |  RQ AGN	|  3   |
        |   FRI     |  4   |
        |   FRII    |  5   |


    Functions
    ---------
    genSpec:
        Generate the spectrum of the source at frequency freq.
    calc_Tb:
        Calculate the average surface brightness, the area of the source
        should be inputed.
    """

    def __init__(self, Freq=150, ClassType=1):

        # Frequency
        self.Freq = Freq
        # ClassType = ClassType
        self.ClassType = ClassType

    def genSpec(self):
        # generate the spectrum
        # Use IF-THEN to replace SWITCH-CASE
        # reference flux at 151MHz, see Willman et al's work
        self.I_151 = 10**(np.random.uniform(-4, -3))
        # Clac flux
        if self.ClassType == 1:
            Spec = (self.Freq / 151e6)**(-0.7) * self.I_151
        elif self.ClassType == 2:
            Spec = (self.Freq / 151e6)**(-0.7) * self.I_151
        elif self.ClassType == 3:
            Spec = (self.Freq / 151e6)**(-0.7) * self.I_151
        elif self.ClassType == 4:
            Spec_lobe = (self.Freq / 151e6)**-0.75 * self.I_151
            a0 = np.log10(self.I_151) - 0.7 * np.log10(151e6) + \
                0.29 * np.log10(151e6) * np.log10(151e6)
            lgs = a0 + 0.7 * np.log10(self.Freq) - 0.29 * \
                np.log10(self.Freq) * np.log10(self.Freq)
            Spec_core = 10**lgs
            Spec = np.array([Spec_core, Spec_lobe])
        elif self.ClassType == 5:
            Spec_lobe = (self.Freq / 151e6)**-0.75 * self.I_151
            Spec_hotspot = (self.Freq / 151e6)**-0.75 * self.I_151
            a0 = np.log10(self.I_151) - 0.7 * np.log10(151e6) + \
                0.29 * np.log10(151e6) * np.log10(151e6)
            lgs = a0 + 0.7 * np.log10(self.Freq) - 0.29 * \
                np.log10(self.Freq) * np.log10(self.Freq)
            Spec_core = 10**lgs
            Spec = np.array([Spec_core, Spec_lobe, Spec_hotspot])

        return Spec

    # calc_Tb
    def calc_Tb(self, area):
        # light speed
        c = 2.99792458e8
        # ?
        kb = 1.38e-23
        # flux in Jy
        flux_in_Jy = self.genSpec()
        Omegab = area  # [sr]

        Sb = flux_in_Jy * 1e-26 / Omegab
        FluxPixel = Sb / 2 / self.Freq / self.Freq * c * c / kb

        return FluxPixel


def read_csv(FileName, FoldName='PS_tables'):
    """
    Read csv format point source files,judge its class type according
    to its name.
    For example, 'PS_Num_YYYYMMDD_HHMMSS.csv'
    Split it by '_'

    Parameters
    ----------
    FileName: str
        Name of the file.

    """

    # Split and judge point source type
    ClassList = ['SF', 'SB', 'RQ', 'FRI', 'FRII']
    ClassName = FileName.split('_')[0]
    ClassType = ClassList.index(ClassName) + 1
    # Read csv
    PS_data = pd.read_csv(FoldName + '/' + FileName)

    return ClassType, PS_data


def calc_flux(ClassType, Freq, PS_data):
    """
    Calculate the flux and surface brightness of the point source.

    Parameters
    ----------
    ClassType: int
        Type of point source
    Freq: float
        Frequency
    PS_data: pandas.core.frame.DataFrame
        Data of the point sources
    """
    # init flux
    PS_flux = Flux(Freq=Freq, ClassType=ClassType)
    # PS_flux_list
    NumPS = PS_data.shape[0]
    if ClassType <= 3:
        PS_flux_list = np.zeros((NumPS,))
        # Iteratively calculate flux
        for i in range(NumPS):
            PS_area = PS_data['Area (sr)'][i]
            PS_flux_list[i] = PS_flux.calc_Tb(PS_area)
    else:
        PS_flux_list = np.zeros((NumPS, 2))
# Iteratively calculate flux
        for i in range(NumPS):
            PS_area = PS_data['Area (sr)'][i]
            PS_flux_list[i, :] = PS_flux.calc_Tb(PS_area)

    return PS_flux_list


def draw_rq(nside, PS_data, Freq):
    """
    Designed to draw the radio quiet AGN

    Parameters
    ----------
    ImgMat: np.ndarray
        Two dimensional matrix, to describe the image
    PS_data: pandas.core.frame.DataFrame
        Data of the point sources
    Freq: float
        Frequency
    """
    # Init
    pix_vec = np.zeros((12*nside**2,))
    # Gen flux list
    PS_flux_list = calc_flux(3, Freq, PS_data)
    # Angle to pix
    pix = hp.ang2pix(nside, PS_data['Theta (deg)'] /
                     180, PS_data['Phi (deg)'] / 180)
    # Gen pix_vec
    pix_vec[pix] += PS_flux_list

    return pix_vec


def draw_cir(nside, PS_data, ClassType, Freq):
    """
    Designed to draw the circular  star forming  and star bursting PS.

    Prameters
    ---------
    nside: int and dyadic
        number of sub pixel in a cell of the healpix structure
    PS_data: pandas.core.frame.DataFrame
        Data of the point sources
    ClassType: int
        Class type of the point soruces
    Freq: float
        Frequency
    """
    # Init
    pix_vec = np.zeros((12*nside**2,))
    # Gen flux list
    PS_flux_list = calc_flux(ClassType, Freq, PS_data)
    #  Iteratively draw the ps
    NumPS = PS_data.shape[0]
    for i in range(NumPS):
        # grid
        PS_radius = PS_data['radius (rad)'][i]  # radius[rad]
        theta = PS_data['Theta (deg)'][i] * au.deg   # theta
        phi = PS_data['Phi (deg)'][i] * au.deg  # phi
        # Fill with circle
        step = PS_radius / 10  # Should be fixed
        # x and y are the differencial rad to the core point at the theta and
        # phi directions.
        x = np.arange(-PS_radius, PS_radius + step, step) * au.rad
        y = np.arange(- PS_radius,  PS_radius + step, step) * au.rad
        for p in range(len(x)):
            for q in range(len(y)):
                if np.sqrt(x[p].value**2 + y[q].value**2) <= PS_radius:
                    x_ang = x[p].to(au.deg) + theta
                    y_ang = y[q].to(au.deg) + phi
                    pix_tmp = hp.ang2pix(
                        nside, x_ang.value / 180, y_ang.value / 180)
                    pix_vec[pix_tmp] += PS_flux_list[i]

    return pix_vec


def draw_lobe(nside, PS_data, ClassType, Freq):
    """
    Designed to draw the elliptical lobes of FRI and FRII

    Prameters
    ---------
    nside: int and dyadic
    PS_data: pandas.core.frame.DataFrame
        Data of the point sources
    ClassType: int
        Class type of the point soruces
    Freq: float
        Frequency

    """
    # Init
    pix_vec = np.zeros((12*nside**2,))
    NumPS = PS_data.shape[0]
    # Gen flux list
    PS_flux_list = calc_flux(ClassType, Freq, PS_data)
    PS_lobe = PS_flux_list[:,1]
    # Iteratively draw ps
    for i in range(NumPS):
        # Parameters
        theta = PS_data['Theta (deg)'][i] * au.deg
        phi = PS_data['Phi (deg)'][i] * au.deg
        lobe_maj = PS_data['lobe_maj (rad)'][i]
        lobe_min = PS_data['lobe_min (rad)'][i]
        lobe_ang = PS_data['lobe_ang (deg)'][i] / 180

        # Lobe1
        lobe1_theta = lobe_maj * np.cos(lobe_ang) * au.rad
        lobe1_theta = theta + lobe1_theta.to(au.deg)
        lobe1_phi = lobe_maj * np.sin(lobe_ang) * au.rad
        lobe1_phi = phi + lobe1_phi.to(au.deg)
        # Focuses
        lobe_c = np.sqrt(lobe_maj**2 - lobe_min**2)  # focus distance
        F1_core_x = lobe_c
        F1_core_y = 0
        F2_core_x = -lobe_c
        F2_core_y = 0
        # draw
        step = lobe_maj / 10
        x = np.arange(-lobe_maj, lobe_maj + step, step)
        y = np.arange(-lobe_min, lobe_min + step, step)
        # Ellipse
        for p in range(len(x)):
            for q in range(len(y)):
                DistFocus1 = np.sqrt(
                    (x[p] - F1_core_x)**2 + (y[q] - F1_core_y)**2)
                DistFocus2 = np.sqrt(
                    (x[p] - F2_core_x)**2 + (y[q] - F2_core_y)**2)
                if (DistFocus1 + DistFocus2 <= 2 * lobe_maj):
                    # rotation
                    x_ang = (x[p] * au.rad).to(au.deg)
                    y_ang = (y[q] * au.rad).to(au.deg)
                    x_r = x_ang * np.cos(lobe_ang) - y_ang * np.sin(lobe_ang)
                    y_r = x_ang * np.sin(lobe_ang) + y_ang * np.cos(lobe_ang)
                    # Judge and Fill
                    pix_tmp = hp.ang2pix(
                        nside, (theta + x_r).value / 180, (phi + y_r).value / 180)
                    pix_vec[pix_tmp] += PS_lobe[i]

        # Lobe2
        lobe2_theta = lobe_maj * np.cos(lobe_ang + np.pi) * au.rad
        lobe2_theta = theta + lobe2_theta.to(au.deg)
        lobe2_phi = lobe_maj * np.sin(lobe_ang + np.pi) * au.rad
        lobe2_phi = phi + lobe2_phi.to(au.deg)
        # Focuses
        lobe_c = np.sqrt(lobe_maj**2 - lobe_min**2)  # focus distance
        F1_core_x = lobe_c
        F1_core_y = 0
        F2_core_x = -lobe_c
        F2_core_y = 0
        # draw
        step = lobe_maj / 10
        x = np.arange(-lobe_maj, lobe_maj + step, step)
        y = np.arange(-lobe_min, lobe_min + step, step)
        # Ellipse
        for p in range(len(x)):
            for q in range(len(y)):
                DistFocus1 = np.sqrt(
                    (x[p] - F1_core_x)**2 + (y[q] - F1_core_y)**2)
                DistFocus2 = np.sqrt(
                    (x[p] - F2_core_x)**2 + (y[q] - F2_core_y)**2)
                if (DistFocus1 + DistFocus2 <= 2 * lobe_maj):
                    # rotation
                    x_ang = (x[p] * au.rad).to(au.deg)
                    y_ang = (y[q] * au.rad).to(au.deg)
                    x_r = x_ang * np.cos(lobe_ang + np.pi) - \
                        y_ang * np.sin(lobe_ang + np.pi)
                    y_r = x_ang * np.sin(lobe_ang + np.pi) + \
                        y_ang * np.cos(lobe_ang + np.pi)
                    # Judge and Fill
                    pix_tmp = hp.ang2pix(
                        nside, (lobe2_theta + x_r).value / 180, (lobe2_phi + y_r).value / 180)
                    pix_vec[pix_tmp] += PS_lobe[i]
        # Core
        pix_tmp = hp.ang2pix(nside, PS_data['Theta (deg)']/ 180, PS_data['Phi (deg)'] / 180)
        PS_core = PS_flux_list[:,0]
        pix_vec[pix_tmp] += PS_core

    return pix_vec


def draw_ps(nside, Freq, FileName, FoldName='PS_tables'):
    """
    Read csv ps list file, and generate the healpix structure vector
    with the respect frequency.

    Prameters
    ---------
    nside: int and dyadic
        Number of subpixel in a healpix cell
    Freq: float
        Frequency
    FileName: str
        Name of the ps list catelogue
    FoldName: str
        Name of the folder saving ps lists, which is 'PS_tables' as default.
    """

    # Init
    pix_vec = np.zeros((12 * nside**2,))
    # load csv
    ClassType, PS_data = read_csv(FileName, FoldName)

    # get sparsed matrix
    if ClassType == 1 or ClassType == 2:
        pix_vec = draw_cir(nside, PS_data, ClassType, Freq)
    elif ClassType == 3:
        pix_vec = draw_rq(nside, PS_data, Freq)
    else:
        pix_vec = draw_lobe(nside, PS_data, ClassType, Freq)

    return pix_vec


def sparse2full(nside, sparse_mat):
    """
    Transform the sparsed mat to full healpix vector

    parameters
    ----------
    nside: int and dyadic
        Number of subpixel in the cell of healpix
    sparse_mat: np.ndarray(2,NumPS)
        The sparsed mat, pix indices are in column1, and
        brightness in column2

    return
    ------
    pix_vec: np.ndarray(1,12*nside^2)
    """
    # init
    pix_vec = np.zeros((12 * nside*nside,))
    # Judge the legality of sparse_mat
    if sparse_mat.shape[1] != 2:
        print("The structure of sparsed matrix is illegal!")
        return pix_vec
    else:
        pix_vec[sparse_mat[:, 0].tolist()] += sparse_mat[:, 1].tolist()
        return pix_vec

def main(argv):
    """
    A main function for use this module at the command window

    parameters
    ---------
    argv: a string array
    argv[0] module name
    argv[1:] some parameters and filenames

    example
    -------
    psDraw_new -i PS_tables/SF_100_20161009_205700.csv -o PS_tables/SF_nside_512.fits
    -n 512 -f 150
    """
    try:
        opts,args = getopt.getopt(argv,"hi:o:n:f:",["infile=","outfile=","nside","freq"])
    except getopt.GetoptError:
        print("pyDraw -i <PS name (csv)> -o <Outpur fits name> -n <nside> -f <frequency>")
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
            print("pyDraw -i <PS name (csv)> -o <Outpur fits name> -n <nside> -f <frequency>")
        elif opt in ("-i","--infile"):
            ps_name = arg
        elif opt in ("-o","--outfile"):
            fits_name = arg
        elif opt in ("-n","--nside"):
            nside = int(arg)
        elif opt in ("-f","--freq"):
            freq = float(arg)
    # Split to get folder name and file name
    Str_split = ps_name.split('/')
    if len(Str_split) == 1:
        FileName = Str_split
        FoldName = "./"
    else:
        FileName = Str_split[-1]
        FoldName = ''
        for i in range(len(Str_split)-1):
            FoldName = FoldName + Str_split[i]
    # print
    print("FoldName: ",FoldName)
    print("FileName: ",FileName)
    print("nside: ",nside)
    print("frequency: ",freq)
    # get pix_vec
    if 'nside' not in dir():
        nside = 512
    if 'freq' not in dir():
        freq = 150
    pix_vec = draw_ps(nside,freq,FileName,FoldName)
    # save
    if 'fits_name' in dir():
        write_fits_healpix(fits_name,pix_vec)
    else:
        fits_name = FoldName + '/PS_nside_' + str(nside) +'_'+str(freq)+ '.fits'
        write_fits_healpix(fits_name,pix_vec)

if __name__ == "__main__":
    main(sys.argv[1:])
