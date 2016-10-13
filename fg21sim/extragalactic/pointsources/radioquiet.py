# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

from .base import BasePointSource

class RadioQuiet(BasePointSource):
    def __init__(self,configs):
        super().__init__(configs)
        self._get_configs()

    def _get_configs(self):
        """Load the configs and set the corresponding class attributes"""

        # point sources amount
        self.num_ps = self.configs.getn("extragalactic/pointsource/num_rq")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsource/prefix_rq")
