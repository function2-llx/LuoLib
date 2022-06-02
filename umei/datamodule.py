from collections.abc import Sequence

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from monai.data import DataLoader, Dataset, select_cross_validation_folds
from .args import UMeIArgs

class CVDataModule(LightningDataModule):
    partitions: list[Sequence]

    def __init__(self, args: UMeIArgs):
        super().__init__()
        self.args = args
        self.val_id = 0

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
        ret = {'val': self.val_id}
        if self.args.use_test_fold:
            ret['test'] = self.args.num_folds - 1
        return ret

    @property
    def train_transform(self):
        raise NotImplementedError

    @property
    def eval_transform(self):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        return DataLoader(
            dataset=Dataset(
                select_cross_validation_folds(
                    self.partitions,
                    folds=np.delete(range(self.num_cv_folds), self.val_id)
                ),
                transform=self.train_transform,
            ),
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        from monai.data import DataLoader, Dataset

        val_ids = list(self.val_parts.values())
        if not all(
            len(self.partitions[val_ids[0]]) == len(self.partitions[val_ids[i]])
            for i in range(1, len(val_ids))
        ):
            import warnings
            warnings.warn(f'length of val{self.val_id} and test folds are not equal')

        return CombinedLoader(
            loaders={
                split: DataLoader(
                    dataset=Dataset(self.partitions[part_id], transform=self.eval_transform),
                    num_workers=self.args.dataloader_num_workers,
                    batch_size=self.args.per_device_eval_batch_size,
                    pin_memory=True,
                    persistent_workers=True,
                )
                for split, part_id in self.val_parts.items()
            },
            mode='max_size_cycle',
        )
