from functools import cached_property
from typing import Callable, Sequence

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset as TorchDataset, RandomSampler

import monai
from monai import transforms as monai_t
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, Dataset, partition_dataset_classes, select_cross_validation_folds

from umei.conf import CrossValConf, ExpConfBase
from umei.utils import DataKey, DataSplit

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

    def load_data_transform(self, stage: DataSplit) -> list[Callable]:
        raise NotImplementedError

    # crop/pad, affine
    def spatial_normalize_transform(self, stage: DataSplit) -> list[Callable]:
        raise NotImplementedError

    def aug_transform(self) -> list[Callable]:
        raise NotImplementedError

    def intensity_normalize_transform(self, _stage) -> list[Callable]:
        conf = self.conf
        transforms = []

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
        return transforms

    def post_transform(self, _stage):
        raise NotImplementedError

    def train_transform(self) -> Callable:
        stage = DataSplit.TRAIN
        return monai.transforms.Compose([
            *self.load_data_transform(stage),
            *self.spatial_normalize_transform(stage),
            *self.aug_transform(),
            *self.intensity_normalize_transform(stage),
            *self.post_transform(stage),
        ])

    def val_transform(self) -> Callable:
        stage = DataSplit.VAL
        return monai.transforms.Compose([
            *self.load_data_transform(stage),
            *self.spatial_normalize_transform(stage),
            *self.intensity_normalize_transform(stage),
            *self.post_transform(stage),
        ])

    def test_transform(self):
        stage = DataSplit.TEST
        return monai.transforms.Compose([
            *self.load_data_transform(stage),
            *self.spatial_normalize_transform(stage),
            *self.intensity_normalize_transform(stage),
            *self.post_transform(stage),
        ])

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
            shuffle=None if conf.max_epochs is None else True,
            sampler=RandomSampler(
                dataset,
                num_samples=conf.max_steps * conf.train_batch_size,
            ) if conf.max_epochs is None else None,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            persistent_workers=True if self.conf.dataloader_num_workers > 0 else False,
        )

    def build_eval_dataloader(self, eval_dataset: TorchDataset):
        conf = self.conf
        device_count = torch.cuda.device_count()
        assert conf.eval_batch_size % torch.cuda.device_count() == 0
        per_device_eval_batch_size = conf.eval_batch_size // device_count
        return DataLoader(
            eval_dataset,
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
    conf: CrossValConf | ExpConfBase

    def __init__(self, conf):
        super().__init__(conf)
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
        # deliberately not using self.conf.num_folds
        fold_ids = range(len(self.partitions))
        if self.val_id is not None:
            fold_ids = np.delete(fold_ids, self.val_id)
        return select_cross_validation_folds(self.partitions, fold_ids)

    def val_data(self):
        if self.val_id is None:
            return []
        return select_cross_validation_folds(self.partitions, folds=self.val_id)

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