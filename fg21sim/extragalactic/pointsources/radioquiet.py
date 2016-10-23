# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
from pandas import DataFrame
from .base import BasePointSource

# Defintion of radio-quiet AGN
class RadioQuiet(BasePointSource):
    def __init__(self,configs):
        BasePointSource.__init__(self,configs)
        self._get_configs()

    def _get_configs(self):
        """ Load the configs and set the corresponding class attributes"""
        # point sources amount
        self.NumPS = self.configs.getn("extragalactic/pointsource/Num_rq")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsource/prefix_rq")
