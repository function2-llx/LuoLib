from collections.abc import Hashable, Sequence
from functools import cached_property
import itertools as it
from typing import Callable

import numpy as np
from torch.utils.data import Dataset as TorchDataset, RandomSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

import monai
from monai import transforms as monai_t
from monai.transforms import AdjustContrast as GammaCorrection
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor, partition_dataset_classes, select_cross_validation_folds
from monai.utils import GridSampleMode, GridSamplePadMode, PytorchPadMode
from .args import AugArgs, CVArgs, SegArgs, UMeIArgs
from .transforms import (
    RandAffineCropD, RandCenterGeneratorByLabelClassesD, RandSpatialCenterGeneratorD,
    SpatialRangeGenerator,
)
from .utils import DataKey, DataSplit

class UMeIDataModule(LightningDataModule):
    def __init__(self, args: UMeIArgs):
        super().__init__()
        self.args = args

    def train_data(self) -> Sequence:
        raise NotImplementedError

    def val_data(self) -> dict[DataSplit, Sequence] | Sequence:
        raise NotImplementedError

    def test_data(self) -> Sequence:
        raise NotImplementedError

    @property
    def train_transform(self):
        raise NotImplementedError

    @property
    def val_transform(self):
        raise NotImplementedError

    @property
    def test_transform(self):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        args = self.args
        dataset = CacheDataset(
            self.train_data(),
            transform=self.train_transform,
            cache_num=self.args.train_cache_num,
            num_workers=self.args.cache_dataset_workers,
        )
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=RandomSampler(
                dataset,
                num_samples=args.per_device_train_batch_size * args.num_epoch_batches,
            ),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

    def build_eval_dataloader(self, dataset: TorchDataset):
        return DataLoader(
            dataset,
            num_workers=self.args.dataloader_num_workers,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

    def val_dataloader(self):
        return self.build_eval_dataloader(CacheDataset(
            self.val_data(),
            transform=self.val_transform,
            cache_num=self.args.val_cache_num,
            num_workers=self.args.cache_dataset_workers,
        ))

    def test_dataloader(self):
        return self.build_eval_dataloader(Dataset(
            self.test_data(),
            transform=self.test_transform,
        ))

class CVDataModule(UMeIDataModule):
    args: CVArgs | UMeIArgs

    def __init__(self, args: CVArgs | UMeIArgs):
        super().__init__(args)
        self.val_id = 0

    # all data for fit (including train & val)
    def fit_data(self) -> tuple[Sequence, Sequence]:
        raise NotImplementedError

    @property
    def val_id(self) -> int:
        return self._val_id

    @val_id.setter
    def val_id(self, x: int):
        assert x in range(self.args.num_folds)
        self._val_id = x

    @cached_property
    def partitions(self):
        fit_data, classes = self.fit_data()
        if classes is None:
            classes = [0] * len(fit_data)
        return partition_dataset_classes(
            fit_data,
            classes,
            num_partitions=self.args.num_folds,
            shuffle=True,
            seed=self.args.seed,
        )

    def train_data(self):
        return select_cross_validation_folds(
            self.partitions,
            folds=np.delete(range(len(self.partitions)), self.val_id),
        )

    def val_data(self):
        return select_cross_validation_folds(self.partitions, folds=self.val_id)

class SegDataModule(UMeIDataModule):
    args: UMeIArgs | SegArgs | AugArgs

    def __init__(self, args: UMeIArgs | SegArgs):
        super().__init__(args)

    def loader_transform(self):
        # def fix_seg_affine(data: dict):
        #     if load_seg:
        #         data[f'{DataKey.SEG}_meta_dict']['affine'] = data[f'{DataKey.IMG}_meta_dict']['affine']
        #     return data
        transforms = [
            monai_t.LoadImageD([DataKey.IMG, DataKey.SEG], ensure_channel_first=True, image_only=True),
        ]
        return transforms

    def normalize_transform(self, *, transform_seg: bool = True) -> list[monai_t.Transform]:
        args = self.args
        transforms = []
        all_keys = [DataKey.IMG]
        if transform_seg:
            all_keys.append(DataKey.SEG)

        if self.args.norm_intensity:
            if args.a_min is not None:
                transforms.append(monai_t.ThresholdIntensityD(
                    DataKey.IMG,
                    threshold=args.a_min,
                    above=True,
                    cval=args.a_min,
                ))
            if args.a_max:
                transforms.append(monai_t.ThresholdIntensityD(
                    DataKey.IMG,
                    threshold=args.a_max,
                    above=False,
                    cval=args.a_max,
                ))
            transforms.extend([
                monai_t.CropForegroundD(all_keys, DataKey.IMG, 'min'),
                monai.transforms.NormalizeIntensityD(
                    DataKey.IMG,
                    args.norm_mean,
                    args.norm_std,
                    non_min=True,
                ),
            ])
        else:
            transforms.append(monai.transforms.ScaleIntensityRangeD(
                DataKey.IMG,
                a_min=self.args.a_min,
                a_max=self.args.a_max,
                b_min=self.args.b_min,
                b_max=self.args.b_max,
                clip=True,
            ))

        transforms.extend([
            monai_t.SpacingD(DataKey.IMG, pixdim=args.spacing, mode=GridSampleMode.BILINEAR),
            monai_t.SpatialPadD(
                DataKey.IMG,
                spatial_size=args.sample_shape,
                mode=PytorchPadMode.CONSTANT,
                pad_min=True,
            )
        ])
        if transform_seg:
            transforms.extend([
                monai_t.SpacingD(
                    DataKey.SEG,
                    pixdim=self.args.spacing,
                    mode=GridSampleMode.NEAREST,
                ),
                monai_t.SpatialPadD(
                    DataKey.SEG,
                    spatial_size=args.sample_shape,
                    mode=PytorchPadMode.CONSTANT,
                )
            ])
        return transforms

    def aug_transform(self) -> list[monai_t.Transform]:
        args = self.args
        return [
            RandAffineCropD(
                [DataKey.IMG, DataKey.SEG],
                args.sample_shape,
                [GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                center_generator=monai_t.OneOf(
                    [
                        RandSpatialCenterGeneratorD(DataKey.IMG, args.sample_shape),
                        RandCenterGeneratorByLabelClassesD(
                            DataKey.SEG,
                            args.sample_shape,
                            [0, *it.repeat(1, args.num_seg_classes - 1)],
                            args.num_seg_classes,
                        )
                    ],
                    args.fg_oversampling_ratio,
                ),
                rotate_generator=SpatialRangeGenerator(
                    ((0, 0), (0, 0), (-np.pi / 6, np.pi / 6)),
                    prob=(0, 0, .5),
                ),
                scale_generator=SpatialRangeGenerator(
                    (0.7, 1.4),
                    prob=.5,
                    repeat=3,
                ),
            ),
            monai_t.RandGaussianNoiseD(
                DataKey.IMG,
                prob=args.gaussian_noise_p,
                std=args.gaussian_noise_std,
            ),
            # gaussian blur: monai_t.RandGaussianSmoothD(),
            monai.transforms.RandScaleIntensityD(DataKey.IMG, factors=args.scale_intensity_factor, prob=args.scale_intensity_p),
            monai.transforms.RandShiftIntensityD(DataKey.IMG, offsets=args.shift_intensity_offset, prob=args.shift_intensity_p),
            # contrast
            # simulate low resolution
            # gamma: monai_t.RandAdjustContrastD(),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=args.flip_p, spatial_axis=2),
            monai.transforms.RandRotate90D([DataKey.IMG, DataKey.SEG], prob=args.rotate_p, max_k=1),
        ]

    @property
    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            *self.loader_transform(),
            *self.normalize_transform(),
            *self.aug_transform(),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
    def val_transform(self) -> Callable:
        return monai.transforms.Compose([
            *self.loader_transform(),
            *self.normalize_transform(),
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
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
