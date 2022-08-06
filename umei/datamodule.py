from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

import monai
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, partition_dataset, select_cross_validation_folds
from monai.utils import GridSampleMode, NumpyPadMode
from .args import AugArgs, SegArgs, UMeIArgs
from .utils import DataKey, DataSplit

class UMeIDataModule(LightningDataModule):
    def __init__(self, args: UMeIArgs):
        super().__init__()
        self.args = args

    def train_data(self) -> Sequence:
        raise NotImplementedError

    def val_data(self) -> dict[DataSplit, Sequence] | Sequence:
        raise NotImplementedError

    @property
    def train_transform(self):
        raise NotImplementedError

    @property
    def val_transform(self):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=CacheDataset(
                self.train_data(),
                transform=self.train_transform,
                cache_num=self.args.train_cache_num,
                num_workers=self.args.dataloader_num_workers,
            ),
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

    def val_dataloader(self):
        val_data = self.val_data()
        if isinstance(val_data, dict):
            return CombinedLoader(
                loaders={
                    split: DataLoader(
                        dataset=CacheDataset(
                            data,
                            transform=self.val_transform,
                            cache_num=self.args.val_cache_num,
                            num_workers=self.args.dataloader_num_workers,
                        ),
                        num_workers=self.args.dataloader_num_workers,
                        batch_size=self.args.per_device_eval_batch_size,
                        pin_memory=True,
                        persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
                    )
                    for split, data in self.val_data().items()
                },
                mode='max_size_cycle',
            )
        else:
            return DataLoader(
                dataset=CacheDataset(
                    val_data,
                    transform=self.val_transform,
                    cache_num=self.args.val_cache_num,
                    num_workers=self.args.dataloader_num_workers,
                ),
                num_workers=self.args.dataloader_num_workers,
                batch_size=self.args.per_device_eval_batch_size,
                pin_memory=True,
                persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
            )

class CVDataModule(UMeIDataModule):
    def __init__(self, args: UMeIArgs):
        super().__init__(args)
        self.val_id = 0

        self.partitions = partition_dataset(
            self.fit_data(),
            num_partitions=args.num_folds,
            shuffle=True,
            seed=args.seed,
        )

    # all data for fit (including train & val)
    def fit_data(self) -> Sequence:
        raise NotImplementedError

    @property
    def val_id(self) -> int:
        return self._val_id

    @val_id.setter
    def val_id(self, x: int):
        assert x in range(self.num_cv_folds)
        self._val_id = x

    @property
    def num_cv_folds(self) -> int:
        return self.args.num_folds - self.args.use_test_fold

    @property
    def val_parts(self) -> dict[str, int]:
        ret = {DataSplit.VAL: self.val_id}
        if self.args.use_test_fold:
            ret[DataSplit.TEST] = self.args.num_folds - 1
        return ret

    def train_data(self):
        return select_cross_validation_folds(
            self.partitions,
            folds=np.delete(range(self.num_cv_folds), self.val_id),
        )

    def val_data(self):
        val_ids = list(self.val_parts.values())
        if not all(
            len(self.partitions[val_ids[0]]) == len(self.partitions[val_ids[i]])
            for i in range(1, len(val_ids))
        ):
            import warnings
            warnings.warn(f'length of val{self.val_id} and test folds are not equal')

        return {
            split: select_cross_validation_folds(
                self.partitions,
                folds=part_id,
            )
            for split, part_id in self.val_parts.items()
        }

class SegDataModule(UMeIDataModule):
    args: UMeIArgs | SegArgs | AugArgs

    def __init__(self, args: UMeIArgs | SegArgs):
        super().__init__(args)

    def loader_transform(self, *, load_seg: bool) -> monai.transforms.Compose:
        load_keys = [DataKey.IMG]
        if load_seg:
            load_keys.append(DataKey.SEG)

        # def fix_seg_affine(data: dict):
        #     if load_seg:
        #         data[f'{DataKey.SEG}_meta_dict']['affine'] = data[f'{DataKey.IMG}_meta_dict']['affine']
        #     return data

        return monai.transforms.Compose([
            monai.transforms.LoadImageD(load_keys),
            # monai.transforms.Lambda(fix_seg_affine),
            monai.transforms.AddChannelD(load_keys),
            monai.transforms.OrientationD(load_keys, axcodes='RAS'),
        ])

    def normalize_transform(self, *, full_seg: bool) -> monai.transforms.Compose:
        spacing_keys = [DataKey.IMG]
        spacing_modes = [GridSampleMode.BILINEAR]
        if not full_seg:
            spacing_keys.append(DataKey.SEG)
            spacing_modes.append(GridSampleMode.NEAREST)
        transforms = [monai.transforms.SpacingD(spacing_keys, pixdim=self.args.spacing, mode=spacing_modes)]
        if self.args.norm_intensity:
            transforms.extend([
                monai.transforms.NormalizeIntensityD(DataKey.IMG),
                monai.transforms.ThresholdIntensityD(DataKey.IMG, threshold=-5, above=True, cval=-5),
                monai.transforms.ThresholdIntensityD(DataKey.IMG, threshold=5, above=False, cval=5),
                monai.transforms.ScaleIntensityD(DataKey.IMG, minv=0, maxv=1),
            ])
        else:
            transforms.append(monai.transforms.ScaleIntensityRanged(
                DataKey.IMG,
                a_min=self.args.a_min,
                a_max=self.args.a_max,
                b_min=self.args.b_min,
                b_max=self.args.b_max,
                clip=True,
            ))
        if not full_seg:
            transforms.append(
                monai.transforms.CropForegroundd([DataKey.IMG, DataKey.SEG], source_key=DataKey.IMG)
            )
        return monai.transforms.Compose(transforms)

    def aug_transform(self) -> monai.transforms.Compose:
        crop_transform = {
            'cls': monai.transforms.RandCropByLabelClassesD(
                [DataKey.IMG, DataKey.SEG],
                label_key=DataKey.SEG,
                spatial_size=self.args.sample_shape,
                num_classes=self.args.num_seg_classes,
                num_samples=self.args.num_crop_samples,
            ),
            'pn': monai.transforms.RandCropByPosNegLabeld(
                keys=[DataKey.IMG, DataKey.SEG],
                label_key=DataKey.SEG,
                spatial_size=self.args.sample_shape,
                pos=1,
                neg=1,
                num_samples=self.args.num_crop_samples,
                image_key=DataKey.IMG,
                image_threshold=0,
            )
        }[self.args.crop]

        return monai.transforms.Compose([
            monai.transforms.SpatialPadD(
                [DataKey.IMG, DataKey.SEG],
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT
            ),
            crop_transform,
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD([DataKey.IMG, DataKey.SEG], prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.RandRotate90D([DataKey.IMG, DataKey.SEG], prob=self.args.rotate_p, max_k=3),
            monai.transforms.RandScaleIntensityD(DataKey.IMG, factors=0.1, prob=self.args.scale_p),
            monai.transforms.RandShiftIntensityD(DataKey.IMG, offsets=0.1, prob=self.args.shift_p),
        ])

    @property
    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            *self.loader_transform(load_seg=True).transforms,
            *self.normalize_transform(full_seg=False).transforms,
            *self.aug_transform().transforms,
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
    def val_transform(self) -> Callable:
        return monai.transforms.Compose([
            *self.loader_transform(load_seg=True).transforms,
            *self.normalize_transform(full_seg=False).transforms,
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
