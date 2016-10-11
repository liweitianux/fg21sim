#!/usr/bin/env python3
#
# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Simulation of point sources for 21cm signal detection

Point sources types
-----
1. Star forming (SF) galaxies
2. Star bursting (SB) galaxies
3. Radio quiet AGN (RQ_AGN)
4. Faranoff-Riley I (FRI)
5. Faranoff-Riley II (FRII)

References
------------
[1] Wilman R J, Miller L, Jarvis M J, et al.
    A semi-empirical simulation of the extragalactic radio continuum sky for
    next generation radio telescopes[J]. Monthly Notices of the Royal
    Astronomical Society, 2008, 388(3):1335–1348.
[2] Jelic et al.
[3] Healpix, healpy.readthedocs.io
"""
# Import packages or modules
# Import packages or modules
import os
# import healpy as hp  # healpy
import numpy as np
import time
from pandas import DataFrame
# import custom modules
import base

# Defintion of radio-quiet AGN
class radioquiet(base.pointsource):
    def __init__(self,nside=512):
        base.pointsource.__init__(self,nside)

    def save_as_csv(self, NumPS=100, folder_name='PS_tables/'):
        """
        Generate NumPS of point sources and save them into a csv file.
        """
        # Init
        PS_Table = np.zeros((NumPS, self.nCols))
        for x in range(NumPS):
            PS_Table[x, :] = self.gen_sgl_ps()

        # Transform into Dataframe
        PS_frame = DataFrame(PS_Table, columns=self.Columns,
                             index=list(range(NumPS)))

        # Save to csv
        if os.path.exists(folder_name) == False:
            os.mkdir(folder_name)

        file_name = 'RQ_' + str(NumPS) + '_' + \
            time.strftime('%Y%m%d_%H%M%S') + '.csv'
        PS_frame.to_csv(folder_name + '/' + file_name)
        return PS_frame
