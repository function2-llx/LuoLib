from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Optional

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

MIN_HU = -1024

class Stoic2021DataModule(CVDataModule):
    args: Stoic2021Args

    def __init__(self, args: Stoic2021Args, predict_case: Optional[dict] = None):
        super().__init__(args)
        if args.on_submit:
            self.predict_case = self.infer_transform(predict_case)
        else:
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

    def test_dataloader(self):
        assert self.args.use_test_fold

        return DataLoader(
            dataset=Dataset(self.partitions[-1], transform=self.eval_transform),
            num_workers=self.args.dataloader_num_workers,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        assert self.args.on_submit

        return DataLoader(
            dataset=Dataset([self.predict_case]),
            batch_size=1,
            pin_memory=True,
        )

    @property
    def stat(self):
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
        return yaml.load(stat_path)

    @property
    def loader_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.LoadImageD([self.args.img_key, self.args.mask_key]),
            monai.transforms.AddChannelD([self.args.img_key, self.args.mask_key]),
            monai.transforms.OrientationD([self.args.img_key, self.args.mask_key], axcodes='RAS'),
        ])

    @property
    def crop_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.ThresholdIntensityD('img', threshold=MIN_HU, above=True, cval=MIN_HU),
            monai.transforms.LambdaD('img', lambda x: x - MIN_HU),
            monai.transforms.MaskIntensityD('img', mask_key='mask'),
            monai.transforms.LambdaD('img', lambda x: x + MIN_HU),
            monai.transforms.CropForegroundD(['img', 'mask'], source_key='mask'),
        ])

    @property
    def normalize_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.NormalizeIntensityD(
                self.args.img_key,
                subtrahend=self.stat['mean'],
                divisor=self.stat['std'],
            ),
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
        item_keys = [self.args.img_key, self.args.clinical_key]
        if not self.args.on_submit:
            item_keys.append(self.args.cls_key)
        return monai.transforms.Compose([
            monai.transforms.ResizeD(
                [self.args.img_key, self.args.mask_key],
                spatial_size=self.args.sample_shape,
                mode=[InterpolateMode.AREA, InterpolateMode.NEAREST],
            ),
            monai.transforms.ConcatItemsD([self.args.img_key, self.args.mask_key], name=self.args.img_key),
            monai.transforms.CastToTypeD([self.args.img_key, self.args.clinical_key], dtype=np.float32),
            monai.transforms.SelectItemsD(item_keys),
        ])

    @property
    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.loader_transform,
            self.normalize_transform,
            self.aug_transform,
            self.input_transform,
        ])

    @property
    def eval_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.loader_transform,
            self.normalize_transform,
            self.input_transform,
        ])

    @property
    def infer_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.loader_transform,
            self.crop_transform,
            self.normalize_transform,
            self.input_transform,
        ])
