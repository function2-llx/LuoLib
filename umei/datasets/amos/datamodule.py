import json
from pathlib import Path
from typing import Callable

import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import default_collate

import monai
from monai.data import DataLoader, Dataset, partition_dataset_classes
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datamodule import CVDataModule

from .args import AmosArgs

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

class AmosDataModule(CVDataModule):
    args: AmosArgs

    @staticmethod
    def load_cohort():
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
                            'img': DATA_DIR / img_path,
                            **({} if seg_path is None else {'seg': DATA_DIR / seg_path if seg_path else None})
                        }
                    })
        for split in ['training', 'test']:
            cohort[split] = list(cohort[split].values())
        return cohort

    def __init__(self, args: AmosArgs):
        super().__init__(args)

        self.cohort = AmosDataModule.load_cohort()
        self.partitions = partition_dataset_classes(
            self.cohort['training'],
            classes=pd.DataFrame.from_records(self.cohort['training'])['modality'],
            num_partitions=args.num_folds,
            shuffle=True,
            seed=args.seed,
        )

    def exclude_test(self, subjects: list[str]):
        subjects = set(subjects)
        self.cohort['test'] = list(filter(
            lambda case: case['subject'] not in subjects,
            self.cohort['test'],
        ))

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=Dataset(self.cohort['test'], transform=self.predict_transform),
            num_workers=self.args.dataloader_num_workers,
            batch_size=1,
            pin_memory=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
            collate_fn=lambda batch: {
                **batch[0],
                'img': default_collate([batch[0]['img']]),
            }
        )

    def loader_transform(self, *, on_predict: bool) -> Callable:
        load_keys = [self.args.img_key]
        if not on_predict:
            load_keys.append(self.args.seg_key)

        def fix_seg_affine(data: dict):
            if not on_predict:
                data[f'{self.args.seg_key}_meta_dict']['affine'] = data[f'{self.args.img_key}_meta_dict']['affine']
            return data

        return monai.transforms.Compose([
            monai.transforms.LoadImageD(load_keys),
            monai.transforms.Lambda(fix_seg_affine),
            monai.transforms.AddChannelD(load_keys),
            monai.transforms.OrientationD(load_keys, axcodes='RAS'),
        ])

    def normalize_transform(self, *, on_predict: bool) -> Callable:
        all_keys = [self.args.img_key]
        spacing_modes = [GridSampleMode.BILINEAR]
        if not on_predict:
            all_keys.append(self.args.seg_key)
            spacing_modes.append(GridSampleMode.NEAREST)
        return monai.transforms.Compose([
            monai.transforms.SpacingD(all_keys, pixdim=self.args.spacing, mode=spacing_modes),
            monai.transforms.NormalizeIntensityD(self.args.img_key),
            monai.transforms.ThresholdIntensityD(self.args.img_key, threshold=-5, above=True, cval=-5),
            monai.transforms.ThresholdIntensityD(self.args.img_key, threshold=5, above=False, cval=5),
            monai.transforms.ScaleIntensityD(self.args.img_key, minv=0, maxv=1),
        ])

    @property
    def aug_transform(self) -> Callable:
        return monai.transforms.Compose([
            monai.transforms.SpatialPadD(
                [self.args.img_key, self.args.seg_key],
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT
            ),
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
    def train_transform(self) -> Callable:
        return monai.transforms.Compose([
            self.loader_transform(on_predict=False),
            self.normalize_transform(on_predict=False),
            self.aug_transform,
            monai.transforms.SelectItemsD([self.args.img_key, self.args.seg_key]),
        ])

    @property
    def eval_transform(self) -> Callable:
        if self.args.use_monai:
            val_transform = monai.transforms.Compose(
                [
                    monai.transforms.LoadImageD([self.args.img_key, self.args.seg_key]),
                    monai.transforms.AddChannelD([self.args.img_key, self.args.seg_key]),
                    monai.transforms.OrientationD([self.args.img_key, self.args.seg_key], axcodes="RAS"),
                    monai.transforms.SpacingD(
                        [self.args.img_key, self.args.seg_key],
                        pixdim=self.args.spacing,
                        mode=("bilinear", "nearest"),
                    ),
                    monai.transforms.ScaleIntensityRanged(
                        keys=self.args.img_key,
                        a_min=-175,
                        a_max=250,
                        b_min=0,
                        b_max=1,
                        clip=True,
                    ),
                    monai.transforms.CropForegroundd([self.args.img_key, self.args.seg_key], source_key=self.args.img_key),
                    monai.transforms.ToTensord([self.args.img_key, self.args.seg_key]),
                ]
            )
            return val_transform
        return monai.transforms.Compose([
            self.loader_transform(on_predict=False),
            self.normalize_transform(on_predict=False),
        ])

    @property
    def predict_transform(self):
        return monai.transforms.Compose([
            self.loader_transform(on_predict=True),
            self.normalize_transform(on_predict=True),
        ])
