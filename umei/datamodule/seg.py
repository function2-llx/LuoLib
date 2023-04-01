import itertools as it

import monai
from monai import transforms as monai_t
from monai.utils import GridSampleMode, PytorchPadMode
from monai.transforms import RandAdjustContrastD as RandGammaCorrectionD

from umei.conf import SegExpConf
from umei.transforms import (
    RandAdjustContrastD, RandAffineCropD, RandCenterGeneratorByLabelClassesD, RandSpatialCenterGeneratorD,
    SimulateLowResolutionD,
)
from umei.utils import DataKey, DataSplit
from .base import ExpDataModuleBase

class SegDataModule(ExpDataModuleBase):
    conf: SegExpConf

    def load_data_transform(self, _stage) -> list:
        return [monai_t.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=True, image_only=True)]

    def spatial_normalize_transform(self, stage: DataSplit):
        conf = self.conf
        transform_keys = [DataKey.IMG]
        if stage in [DataSplit.TRAIN, DataSplit.VAL]:
            transform_keys.append(DataKey.SEG)
        transforms = [monai_t.CropForegroundD(transform_keys, DataKey.IMG, 'min')]
        if conf.spacing is not None:
            transforms.append(monai_t.SpacingD(DataKey.IMG, pixdim=conf.spacing, mode=GridSampleMode.BILINEAR))
        transforms.append(
            monai_t.SpatialPadD(
                DataKey.IMG,
                spatial_size=conf.sample_shape,
                mode=PytorchPadMode.CONSTANT,
                pad_min=True,
            )
        )
        if DataKey.SEG in transform_keys:
            if conf.spacing is not None:
                transforms.append(monai_t.SpacingD(DataKey.SEG, pixdim=conf.spacing, mode=GridSampleMode.NEAREST))
            transforms.append(
                monai_t.SpatialPadD(
                    DataKey.SEG,
                    spatial_size=conf.sample_shape,
                    mode=PytorchPadMode.CONSTANT,
                )
            )
        return transforms

    def aug_transform(self):
        conf = self.conf
        return [
            RandAffineCropD(
                [DataKey.IMG, DataKey.SEG],
                conf.sample_shape,
                [GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                conf.rotate_range,
                conf.rotate_p,
                conf.scale_range,
                conf.scale_p,
                conf.spatial_dims,
                conf.dummy_dim,
                center_generator=monai_t.OneOf(
                    [
                        RandSpatialCenterGeneratorD(DataKey.IMG, conf.sample_shape),
                        RandCenterGeneratorByLabelClassesD(
                            DataKey.SEG,
                            conf.sample_shape,
                            [0, *it.repeat(1, conf.num_seg_classes - 1)],
                            conf.num_seg_classes,
                        )
                    ],
                    conf.fg_oversampling_ratio,
                ),
            ),
            monai_t.RandGaussianNoiseD(
                DataKey.IMG,
                prob=conf.gaussian_noise_p,
                std=conf.gaussian_noise_std,
            ),
            monai_t.RandGaussianSmoothD(
                DataKey.IMG,
                conf.gaussian_smooth_std_range,
                conf.gaussian_smooth_std_range,
                conf.gaussian_smooth_std_range,
                prob=conf.gaussian_smooth_p,
                isotropic_prob=conf.gaussian_smooth_isotropic_prob,
            ),
            monai.transforms.RandScaleIntensityD(DataKey.IMG, factors=conf.scale_intensity_factor, prob=conf.scale_intensity_p),
            monai.transforms.RandShiftIntensityD(DataKey.IMG, offsets=conf.shift_intensity_offset, prob=conf.shift_intensity_p),
            RandAdjustContrastD(DataKey.IMG, conf.adjust_contrast_range, conf.adjust_contrast_p),
            SimulateLowResolutionD(DataKey.IMG, conf.simulate_low_res_zoom_range, conf.simulate_low_res_p, conf.dummy_dim),
            RandGammaCorrectionD(DataKey.IMG, conf.gamma_p, conf.gamma_range),
            *[
                monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=conf.flip_p, spatial_axis=i)
                for i in range(conf.spatial_dims)
            ],
        ]

    def post_transform(self, _stage):
        return [monai_t.SelectItemsD([DataKey.IMG, DataKey.SEG])]
