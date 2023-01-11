from collections.abc import Hashable, Sequence
from functools import cached_property
from typing import Callable

import numpy as np
from torch.utils.data import Dataset as TorchDataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

import monai
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor, partition_dataset_classes, select_cross_validation_folds
from monai.utils import GridSampleMode, NumpyPadMode
from .args import AugArgs, CVArgs, SegArgs, UMeIArgs
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
        return DataLoader(
            dataset=CacheDataset(
                self.train_data(),
                transform=self.train_transform,
                cache_num=self.args.train_cache_num,
                num_workers=self.args.cache_dataset_workers,
            ),
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
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
        transforms = [monai.transforms.SpacingD(DataKey.IMG, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR)]

        def space_seg(data: dict[Hashable, MetaTensor]):
            from batchgenerators.augmentations.utils import resize_segmentation
            img = data[DataKey.IMG]
            seg = data[DataKey.SEG]
            seg.array = resize_segmentation(seg.array, data[DataKey.IMG].shape, order=1)
            seg.affine = img.affine
            return data

        if not full_seg:
            if self.args.spline_seg:
                transforms.append(monai.transforms.Lambda(space_seg))
            else:
                transforms.append(monai.transforms.SpacingD(
                    DataKey.SEG,
                    pixdim=self.args.spacing,
                    mode=GridSampleMode.NEAREST,
                ))
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
            *self.normalize_transform(full_seg=True).transforms,
            monai.transforms.SelectItemsD([DataKey.IMG, DataKey.SEG]),
        ])

    @property
    def test_transform(self):
        return monai.transforms.Compose([
            *self.loader_transform(load_seg=True).transforms,
            *self.normalize_transform(full_seg=True).transforms,
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
