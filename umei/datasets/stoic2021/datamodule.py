from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from ruamel.yaml import YAML

import monai
from monai.data import DataLoader, Dataset, DatasetSummary, partition_dataset_classes, select_cross_validation_folds
from monai.utils import InterpolateMode, NumpyPadMode

import umei
from umei.utils import CVDataModule
from .args import Stoic2021Args

yaml = YAML()
DATASET_ROOT = Path(__file__).parent

class Stoic2021DataModule(CVDataModule):
    args: Stoic2021Args

    def __init__(self, args: Stoic2021Args):
        super().__init__(args)
        if not args.on_submit:
            ref = pd.read_csv(DATASET_ROOT / 'reference.csv')
            assert len(ref[(ref['probCOVID'] == 0) & (ref['probSevere'] == 1)]) == 0
            self.train_cohort = [
                {
                    args.img_key: DATASET_ROOT / 'cropped' / f'{patient_id}.nii.gz',
                    args.mask_key: DATASET_ROOT / 'cropped' / f'{patient_id}_mask.nii.gz',
                    args.cls_key: pcr + severe,
                    args.clinical_key: np.array([age / 100, sex, sex ^ 1]),
                }
                for _, (patient_id, pcr, severe, age, sex) in ref.iterrows()
            ]

            self.partitions = partition_dataset_classes(
                self.train_cohort,
                classes=pd.DataFrame.from_records(self.train_cohort)[args.cls_key],
                num_partitions=args.num_folds,
                shuffle=True,
                seed=args.seed,
            )
        # print(sum(x[args.clinical_key][0] * 100 for x in self.train_cohort) / len(self.train_cohort))
        # for i, part in enumerate(self.partitions):
        #     print(i, sum(x[args.clinical_key][0] * 100 for x in part) / len(part))

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

    def test_dataloader(self):
        assert self.args.use_test_fold

        return DataLoader(
            dataset=Dataset(self.partitions[-1], transform=self.eval_transform),
            num_workers=self.args.dataloader_num_workers,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    def preprocess_transform(self) -> Callable:
        stat_path = DATASET_ROOT / 'stat.yml'
        if not stat_path.exists():
            summary = DatasetSummary(
                Dataset(self.train_cohort, transform=monai.transforms.LoadImageD([self.args.img_key, self.args.mask_key])),
                image_key=self.args.img_key,
                label_key=self.args.mask_key,
                num_workers=self.args.dataloader_num_workers,
            )
            summary.calculate_statistics()
            print(summary.data_mean)
            print(summary.data_std)
            yaml.dump({'mean': summary.data_mean, 'std': summary.data_std}, stat_path)
        stat = yaml.load(stat_path)

        return monai.transforms.Compose([
            monai.transforms.LoadImageD([self.args.img_key, self.args.mask_key]),
            monai.transforms.AddChannelD([self.args.img_key, self.args.mask_key]),
            monai.transforms.OrientationD([self.args.img_key, self.args.mask_key], axcodes='RAS'),
            monai.transforms.NormalizeIntensityD(self.args.img_key, subtrahend=stat['mean'], divisor=stat['std']),
            umei.transforms.SpatialSquarePadD([self.args.img_key, self.args.mask_key], mode=NumpyPadMode.EDGE),
        ])

    @property
    def aug_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.RandFlipD([self.args.img_key, self.args.mask_key], prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipD([self.args.img_key, self.args.mask_key], prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipD([self.args.img_key, self.args.mask_key], prob=0.5, spatial_axis=2),
            monai.transforms.RandRotate90D(
                [self.args.img_key, self.args.mask_key],
                prob=0.5,
                max_k=1,
                spatial_axes=(0, 1)
            ),
        ])

    @property
    def input_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.ResizeD(
                [self.args.img_key, self.args.mask_key],
                spatial_size=[self.args.sample_size, self.args.sample_size, self.args.sample_slices],
                mode=[InterpolateMode.AREA, InterpolateMode.NEAREST],
            ),
            monai.transforms.ConcatItemsD([self.args.img_key, self.args.mask_key], name=self.args.img_key),
            monai.transforms.CastToTypeD([self.args.img_key, self.args.clinical_key], dtype=np.float32),
            monai.transforms.SelectItemsD([self.args.img_key, self.args.clinical_key, self.args.cls_key]),
        ])

    @property
    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.preprocess_transform,
            self.aug_transform,
            self.input_transform,
        ])

    @property
    def eval_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.preprocess_transform,
            self.input_transform,
        ])
