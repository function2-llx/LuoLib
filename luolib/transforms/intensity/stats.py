from collections.abc import Hashable, Mapping

from monai import transforms as mt
from monai.data import MetaTensor

class CleverStatsD(mt.MapTransform):
    def __call__(self, data: Mapping[Hashable, ...]):
        data = dict(data)
        for key in self.key_iterator(data):
            x: MetaTensor = data[key]
            mean = x.new_empty((x.shape[0], ))
            std = x.new_empty((x.shape[0], ))
            for i, v in enumerate(x):
                v = v[v > 0]
                mean[i] = v.mean()
                std[i] = v.std()
            x.meta['mean'] = mean
            x.meta['std'] = std
        return data
