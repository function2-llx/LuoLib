from collections.abc import Sequence
from functools import cached_property
import itertools as it
from typing import Callable

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset as TorchDataset, RandomSampler

import monai
from monai import transforms as monai_t
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, Dataset, partition_dataset_classes, select_cross_validation_folds
from monai.utils import GridSampleMode, PytorchPadMode

from .transforms import RandAdjustContrastD, RandGammaCorrectionD, SimulateLowResolutionD
from .conf import CrossValConf, ExpConfBase, SegExpConf
from .transforms import (
    RandAffineCropD, RandCenterGeneratorByLabelClassesD, RandSpatialCenterGeneratorD,
    SpatialRangeGenerator,
)
from .utils import DataKey, DataSplit

class ExpDataModuleBase(LightningDataModule):
    def __init__(self, conf: ExpConfBase):
        super().__init__()
        self.conf = conf

    def train_data(self) -> Sequence:
        raise NotImplementedError

    def val_data(self) -> dict[DataSplit, Sequence] | Sequence:
        raise NotImplementedError

    def test_data(self) -> Sequence:
        raise NotImplementedError

    def train_transform(self):
        raise NotImplementedError

    def val_transform(self):
        raise NotImplementedError

    def test_transform(self):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        conf = self.conf
        dataset = CacheDataset(
            self.train_data(),
            transform=self.train_transform(),
            cache_num=self.conf.train_cache_num,
            num_workers=self.conf.num_cache_workers,
        )
        device_count = torch.cuda.device_count()
        assert conf.train_batch_size % torch.cuda.device_count() == 0
        per_device_train_batch_size = conf.train_batch_size // device_count
        return DataLoader(
            dataset,
            batch_size=per_device_train_batch_size,
            sampler=RandomSampler(
                dataset,
                num_samples=conf.train_batch_size * conf.num_epoch_batches,
            ),
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            persistent_workers=True if self.conf.dataloader_num_workers > 0 else False,
        )

    def build_eval_dataloader(self, dataset: TorchDataset):
        conf = self.conf
        device_count = torch.cuda.device_count()
        assert conf.eval_batch_size % torch.cuda.device_count() == 0
        per_device_eval_batch_size = conf.eval_batch_size // device_count
        return DataLoader(
            dataset,
            num_workers=self.conf.dataloader_num_workers,
            batch_size=per_device_eval_batch_size,
            pin_memory=self.conf.dataloader_pin_memory,
            persistent_workers=True if self.conf.dataloader_num_workers > 0 else False,
        )

    def val_dataloader(self):
        return self.build_eval_dataloader(CacheDataset(
            self.val_data(),
            transform=self.val_transform(),
            cache_num=self.conf.val_cache_num,
            num_workers=self.conf.num_cache_workers,
        ))

    def test_dataloader(self):
        return self.build_eval_dataloader(Dataset(
            self.test_data(),
            transform=self.test_transform(),
        ))

class CrossValDataModule(ExpDataModuleBase):
    conf: ExpConfBase | CrossValConf

    def __init__(self, conf: ExpConfBase | CrossValConf):
        super().__init__(conf)
        self.conf = conf
        self.val_id = None

    # all data for fit (including train & val)
    def fit_data(self) -> tuple[Sequence, Sequence]:
        raise NotImplementedError

    @property
    def val_id(self) -> int | None:
        return self._val_id

    @val_id.setter
    def val_id(self, x: int | None):
        assert x is None or 0 <= x < self.conf.num_folds
        self._val_id = x

    @cached_property
    def partitions(self):
        fit_data, classes = self.fit_data()
        if classes is None:
            classes = [0] * len(fit_data)
        return partition_dataset_classes(
            fit_data,
            classes,
            num_partitions=self.conf.num_folds,
            shuffle=True,
            seed=self.conf.seed,
        )

    def train_data(self):
        folds = range(len(self.partitions))
        if self.val_id is not None:
            folds = np.delete(folds, self.val_id)
        return select_cross_validation_folds(self.partitions, folds)

    def val_data(self):
        if self.val_id is None:
            return []
        return select_cross_validation_folds(self.partitions, folds=self.val_id)


class SegDataModule(ExpDataModuleBase):
    conf: SegExpConf

    def loader_transform(self):
        transforms = [
            monai_t.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=True, image_only=True),
        ]
        return transforms

    def normalize_transform(self, *, transform_seg: bool = True) -> list[monai_t.Transform]:
        conf = self.conf
        transforms = []
        all_keys = [DataKey.IMG]
        if transform_seg:
            all_keys.append(DataKey.SEG)

        if conf.norm_intensity:
            if conf.intensity_min is not None:
                transforms.append(monai_t.ThresholdIntensityD(
                    DataKey.IMG,
                    threshold=conf.intensity_min,
                    above=True,
                    cval=conf.intensity_min,
                ))
            if conf.intensity_max:
                transforms.append(monai_t.ThresholdIntensityD(
                    DataKey.IMG,
                    threshold=conf.intensity_max,
                    above=False,
                    cval=conf.intensity_max,
                ))
            transforms.extend([
                monai_t.CropForegroundD(all_keys, DataKey.IMG, 'min'),
                monai.transforms.NormalizeIntensityD(
                    DataKey.IMG,
                    conf.norm_mean,
                    conf.norm_std,
                    non_min=True,
                ),
            ])
        else:
            transforms.append(monai.transforms.ScaleIntensityRangeD(
                DataKey.IMG,
                a_min=conf.intensity_min,
                a_max=conf.intensity_max,
                b_min=conf.scaled_intensity_min,
                b_max=conf.scaled_intensity_max,
                clip=True,
            ))

        transforms.extend([
            monai_t.SpacingD(DataKey.IMG, pixdim=conf.spacing, mode=GridSampleMode.BILINEAR),
            monai_t.SpatialPadD(
                DataKey.IMG,
                spatial_size=conf.sample_shape,
                mode=PytorchPadMode.CONSTANT,
                pad_min=True,
            )
        ])
        if transform_seg:
            transforms.extend([
                monai_t.SpacingD(
                    DataKey.SEG,
                    pixdim=conf.spacing,
                    mode=GridSampleMode.NEAREST,
                ),
                monai_t.SpatialPadD(
                    DataKey.SEG,
                    spatial_size=conf.sample_shape,
                    mode=PytorchPadMode.CONSTANT,
                )
            ])
        return transforms

    def aug_transform(self) -> list[monai_t.Transform]:
        conf = self.conf
        return [
            RandAffineCropD(
                [DataKey.IMG, DataKey.SEG],
                conf.sample_shape,
                [GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                dummy_dim=conf.dummy_dim,
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
                rotate_generator=SpatialRangeGenerator(
                    conf.rotate_range,
                    conf.rotate_p,
                    repeat=3 if conf.dummy_dim is None else 1,
                ),
                scale_generator=SpatialRangeGenerator(
                    conf.scale_range,
                    conf.scale_p,
                    default=1.,
                    repeat=3 if conf.dummy_dim is None else 2,
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
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=conf.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=conf.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=conf.flip_p, spatial_axis=2),
        ]

    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            *self.loader_transform(),
            *self.normalize_transform(),
            *self.aug_transform(),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    def val_transform(self) -> Callable:
        return monai.transforms.Compose([
            *self.loader_transform(),
            *self.normalize_transform(),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    def test_transform(self):
        return monai.transforms.Compose([
            *self.loader_transform(),
            *self.normalize_transform(transform_seg=False),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

def load_decathlon_datalist(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: PathLike = None,
):
    from monai.data import load_decathlon_datalist as monai_load
    data = monai_load(data_list_file_path, is_segmentation, data_list_key, base_dir)
    for item in data:
        for data_key, decathlon_key in [
            (DataKey.IMG, 'image'),
            (DataKey.SEG, 'label'),
        ]:
            item[data_key] = item.pop(decathlon_key)
    return data
