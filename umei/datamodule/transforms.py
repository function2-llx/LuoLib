from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from monai import transforms as monai_t
from umei.utils import DataKey

class ImageNetNormalizeMixin:
    def intensity_normalize_transform(self, _stage):
        return [
            monai_t.ScaleIntensityRangeD(DataKey.IMG, 0, 255),
            monai_t.NormalizeIntensityD(
                DataKey.IMG,
                subtrahend=IMAGENET_DEFAULT_MEAN,
                divisor=IMAGENET_DEFAULT_STD,
                channel_wise=True,
            ),
        ]
