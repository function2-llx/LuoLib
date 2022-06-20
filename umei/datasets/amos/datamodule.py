import json
from pathlib import Path
from typing import Callable

import pandas as pd

import monai
from monai.data import partition_dataset_classes
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datamodule import CVDataModule

from .args import AmosArgs

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort(args: AmosArgs):
    cohort = {
        'training': {},
        'test': {}
    }
    # 1: MRI, 0: CT
    for modality, task in [(1, 2), (0, 1)]:
        with open(DATA_DIR / f'task{task}_dataset.json') as f:
            task = json.load(f)
        for split in ['training', 'test']:
            for case in task[split]:
                if split == 'training':
                    img_path = Path(case['image'])
                    seg_path = Path(case['label'])
                else:
                    img_path = Path(case)
                    seg_path = None
                subject = img_path.name[:-7]
                cohort[split].update({
                    subject: {
                        'subject': subject,
                        'modality': modality,
                        args.img_key: DATA_DIR / img_path,
                        args.seg_key: DATA_DIR / seg_path if seg_path else None,
                    }
                })
    for split in ['training', 'test']:
        cohort[split] = list(cohort[split].values())
    return cohort

class AmosDataModule(CVDataModule):
    args: AmosArgs

    def __init__(self, args: AmosArgs):
        super().__init__(args)

        self.cohort = load_cohort(args)
        self.partitions = partition_dataset_classes(
            self.cohort['training'],
            classes=pd.DataFrame.from_records(self.cohort['training'])['modality'],
            num_partitions=args.num_folds,
            shuffle=True,
            seed=args.seed,
        )

    @property
    def loader_transform(self) -> Callable:
        load_keys = [self.args.img_key, self.args.seg_key]
        def fix_seg_affine(data: dict):
            data[f'{self.args.seg_key}_meta_dict']['affine'] = data[f'{self.args.img_key}_meta_dict']['affine']
            return data

        return monai.transforms.Compose([
            monai.transforms.LoadImageD(load_keys),
            monai.transforms.Lambda(fix_seg_affine),
            monai.transforms.AddChannelD(load_keys),
            monai.transforms.OrientationD(load_keys, axcodes='RAS'),
        ])

    @property
    def normalize_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.SpacingD(
                [self.args.img_key, self.args.seg_key],
                pixdim=self.args.spacing,
                mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
            ),
            monai.transforms.NormalizeIntensityD(self.args.img_key),
            monai.transforms.ThresholdIntensityD(self.args.img_key, threshold=-5, above=True, cval=-5),
            monai.transforms.ThresholdIntensityD(self.args.img_key, threshold=5, above=False, cval=5),
            monai.transforms.ScaleIntensityD(self.args.img_key, minv=0, maxv=1),
            monai.transforms.SpatialPadD(
                [self.args.img_key, self.args.seg_key],
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT,
            ),
        ])

    @property
    def aug_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.RandCropByLabelClassesD(
                [self.args.img_key, self.args.seg_key],
                label_key=self.args.seg_key,
                spatial_size=self.args.sample_shape,
                num_classes=self.args.num_seg_classes,
                num_samples=self.args.num_crop_samples,
            ),
            monai.transforms.RandFlipD([self.args.img_key, self.args.seg_key], prob=0.2, spatial_axis=0),
            monai.transforms.RandFlipD([self.args.img_key, self.args.seg_key], prob=0.2, spatial_axis=1),
            monai.transforms.RandFlipD([self.args.img_key, self.args.seg_key], prob=0.2, spatial_axis=2),
            monai.transforms.RandRotate90D([self.args.img_key, self.args.seg_key], prob=0.2, max_k=3),
            monai.transforms.RandScaleIntensityD(self.args.img_key, factors=0.1, prob=0.1),
            monai.transforms.RandShiftIntensityD(self.args.img_key, offsets=0.1, prob=0.1),
        ])

    @property
    def input_transform(self) -> Callable:
        item_keys = [self.args.img_key]
        if not self.args.on_submit:
            item_keys.append(self.args.seg_key)
        return monai.transforms.Compose([
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