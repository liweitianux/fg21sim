# Copyright (c) 2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Virial radius/mass utilities.

.. [duffy2008]
   Duffy et al. 2008, MNRAS, 390, L64
   http://adsabs.harvard.edu/abs/2008MNRAS.390L..64D

.. [ettori2009]
   Ettori & Balestra 2009, A&A, 496, 343
   http://adsabs.harvard.edu/abs/2009A%26A...496..343

.. [lokas2001]
   Łokas & Mamon 2001, MNRAS, 321, 155
   http://adsabs.harvard.edu/abs/2001MNRAS.321..155L
"""

import logging

import numpy as np
from ..share import COSMO


logger = logging.getLogger(__name__)


def concentration(mass, z=0):
    """
    Calculate the NFW concentration parameter (c = r_vir / r_s) from the
    mass-concentration relation: c = A [M/M_pivot]^B (1+z)^C

    Reference: [duffy2008],Tab.1
    """
    A, B, C = 5.71, -0.084, -0.47  # for M200
    M_pivot = 2e12 / COSMO.h  # [Msun]
    return A * (mass/M_pivot) ** B + (1+z) ** C


def mass_fraction_nfw(s, c):
    """
    Calculate the mass fraction (of virial mass) within the scaled radius
    ``s = r / r_vir``.

    Reference: [lokas2001]

    Parameters
    ----------
    s : float
        The radius scaled by the virial radius, i.e., s = r / r_vir.
    c : float
        The concentration parameter, i.e., c = r_vir / r_s.
    """
    nom = np.log(1 + c*s) - c*s / (1 + c*s)
    denom = np.log(1 + c) - c / (1 + c)
    return nom / denom


def M200(M500, z=0):
    """
    Estimate the M200 from the M500.
    """
    # r500 ~= 0.65 r200; Ref.[ettori2009],Sec.2
    s = 0.65
    # Begin with 'M200 = 1.5*M500'
    c = concentration(M500*1.5, z)
    mass = M500 / mass_fraction_nfw(s, c)
    # Again with the updated mass
    c = concentration(mass, z)
    return M500 / mass_fraction_nfw(s, c)
