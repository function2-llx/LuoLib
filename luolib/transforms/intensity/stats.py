from collections.abc import Hashable, Mapping

from monai import transforms as mt
from monai.data import MetaTensor

class CleverStatsD(mt.MapTransform):
    def __call__(self, data: Mapping[Hashable, ...]):
        data = dict(data)
        for key in self.key_iterator(data):
            x: MetaTensor = data[key]
            v = x[x > 0]
            x.meta['stats'] = {
                'mean': v.mean(),
                'std': v.std(),
            }
        return data
