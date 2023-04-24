import itertools as it

from pytorch_lightning.trainer.states import RunningStage

from monai import transforms as monai_t
from monai.utils import GridSampleMode, PytorchPadMode

from luolib.conf import SegExpConf
from luolib.transforms import SimulateLowResolutionD, RandAdjustContrastD, RandAffineCropD, RandGammaCorrectionD
from luolib.transforms.utils import RandCenterGeneratorByLabelClassesD, RandSpatialCenterGeneratorD
from luolib.utils import DataKey
from .base import ExpDataModuleBase

class SegDataModule(ExpDataModuleBase):
    conf: SegExpConf

    def load_data_transform(self, stage: RunningStage) -> list:
        match stage:
            case RunningStage.PREDICTING:
                return [monai_t.LoadImageD(DataKey.IMG, ensure_channel_first=True, image_only=True)]
            case _:
                return [monai_t.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=True, image_only=True)]

    def spatial_normalize_transform(self, stage: RunningStage):
        conf = self.conf
        transform_keys = [DataKey.IMG]
        if stage in [RunningStage.TRAINING, RunningStage.VALIDATING]:
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
                            list(it.repeat(1, conf.num_seg_classes)) if conf.multi_label
                            else [0, *it.repeat(1, conf.num_seg_classes - 1)],
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
            monai_t.RandScaleIntensityD(DataKey.IMG, factors=conf.scale_intensity_factor, prob=conf.scale_intensity_p),
            monai_t.RandShiftIntensityD(DataKey.IMG, offsets=conf.shift_intensity_offset, prob=conf.shift_intensity_p),
            RandAdjustContrastD(DataKey.IMG, conf.adjust_contrast_range, conf.adjust_contrast_p),
            SimulateLowResolutionD(DataKey.IMG, conf.simulate_low_res_zoom_range, conf.simulate_low_res_p, conf.dummy_dim),
            RandGammaCorrectionD(DataKey.IMG, conf.gamma_p, conf.gamma_range),
            *[
                monai_t.RandFlipD([DataKey.IMG, DataKey.SEG], prob=conf.flip_p, spatial_axis=i)
                for i in range(conf.spatial_dims)
            ],
        ]

    def post_transform(self, stage: RunningStage):
        match stage:
            case RunningStage.TRAINING:
                return [monai_t.SelectItemsD([DataKey.IMG, DataKey.SEG])]
            case RunningStage.PREDICTING:
                return [monai_t.SelectItemsD([DataKey.IMG, DataKey.CASE])]
            case _:
                return [monai_t.SelectItemsD([DataKey.IMG, DataKey.SEG, DataKey.CASE])]
