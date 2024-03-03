from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Sequence, final

import cytoolz
from lightning import LightningDataModule
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import Dataset as TorchDataset, RandomSampler

from luolib.utils import DataKey
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, Dataset

@dataclass
class CacheDatasetConf:
    num_workers: int = 8
    train_num: int = 200
    val_num: int = 100

@dataclass(kw_only=True)
class DataLoaderConf:
    train_batch_size: int
    val_batch_size: int = 1
    num_batches: int
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2

    def __post_init__(self):
        if self.num_workers == 0:
            self.persistent_workers = False
            self.prefetch_factor = None

class ExpDataModuleBase(LightningDataModule):
    def __init__(self, cache_dataset: CacheDatasetConf, dataloader: DataLoaderConf):
        super().__init__()
        self.cache_dataset_conf = cache_dataset
        self.dataloader_conf = dataloader

    def train_data(self) -> Sequence:
        raise NotImplementedError

    def train_transform(self) -> Callable:
        raise NotImplementedError

    def train_dataset(self):
        return CacheDataset(
            self.train_data(),
            transform=self.train_transform(),
            cache_num=self.cache_dataset_conf.train_num,
            num_workers=self.cache_dataset_conf.num_workers,
        )

    def get_train_collate_fn(self):
        # TODO: investigate whether `self` will be copied to worker processes and the memory consumption
        from luolib.data.utils import list_data_collate
        return list_data_collate

    @property
    def world_size(self):
        return 1 if self.trainer is None else self.trainer.world_size

    def train_dataloader(self):
        dataset = self.train_dataset()
        conf = self.dataloader_conf
        sampler = RandomSampler(
            # the only useful information provided by `data_source` is its length
            range(len(dataset)),
            replacement=False,
            num_samples=conf.num_batches * conf.train_batch_size * self.world_size,
        )
        if self.world_size > 1:
            # TODO: make this lazy (_DatasetSamplerWrapper)
            sampler = DistributedSamplerWrapper(
                sampler,
                num_replicas=self.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
            )
        return DataLoader(
            dataset,
            batch_size=conf.train_batch_size,
            sampler=sampler,
            num_workers=conf.num_workers,
            pin_memory=conf.pin_memory,
            prefetch_factor=conf.prefetch_factor,
            persistent_workers=conf.persistent_workers,
            collate_fn=self.get_train_collate_fn(),
        )

    def build_eval_dataloader(self, dataset: TorchDataset, batch_size: int):
        conf = self.dataloader_conf
        return DataLoader(
            dataset,
            num_workers=conf.num_workers,
            batch_size=batch_size,
            pin_memory=conf.pin_memory,
            persistent_workers=conf.persistent_workers,
        )

    def val_data(self) -> Sequence:
        raise NotImplementedError

    def val_transform(self) -> Callable:
        raise NotImplementedError

    def val_dataset(self):
        return CacheDataset(
            self.val_data(),
            transform=self.val_transform(),
            cache_num=self.cache_dataset_conf.val_num,
            num_workers=self.cache_dataset_conf.num_workers,
        )

    def val_dataloader(self):
        return self.build_eval_dataloader(self.val_dataset(), self.dataloader_conf.val_batch_size)

    def test_data(self) -> Sequence:
        raise NotImplementedError

    def predict_data(self) -> Sequence:
        raise NotImplementedError

    def predict_transform(self) -> Callable:
        raise NotImplementedError

    def predict_dataset(self):
        return Dataset(self.predict_data(), self.predict_transform())

    def predict_dataloader(self):
        return self.build_eval_dataloader(self.predict_dataset(), 1)

class CrossValDataModule(ExpDataModuleBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_id = None

    def fit_data(self) -> Sequence | Mapping[Hashable, ...]:
        raise NotImplementedError

    @cached_property
    @final
    def _fit_data(self):
        return self.fit_data()

    def splits(self) -> Sequence[tuple[Sequence[Hashable], Sequence[Hashable]]]:
        """
        Returns:
            a sequence of folds, each fold is a tuple of (train keys, val keys)
        """
        raise NotImplementedError

    @cached_property
    @final
    def _splits(self):
        return self.splits()

    @property
    def num_splits(self):
        return len(self._splits)

    @property
    def fold_id(self) -> int | None:
        return self._fold_id

    @fold_id.setter
    def fold_id(self, x: int | None):
        assert x is None or 0 <= x < self.num_splits
        self._fold_id = x

    def _get_data(self, is_val: int):
        keys = self._splits[self.fold_id][is_val]
        return cytoolz.get(keys, self._fit_data)

    def train_data(self):
        return self._get_data(0)

    def val_data(self):
        return self._get_data(1)

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
