from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import default_collate

import monai
from monai.data import CacheDataset, DataLoader, Dataset, partition_dataset_classes
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datamodule import CVDataModule
from umei.swin_mae import SwinMAEArgs
from .args import AmosArgs

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort(task_id: int):
    cohort = {
        'training': {},
        'test': {}
    }
    # 1: MRI, 0: CT
    for modality, task in [(1, 2), (0, 1)]:
        if modality == 1 and task_id == 1:
            continue
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

class AmosDataModule(CVDataModule):
    args: AmosArgs

    def __init__(self, args: AmosArgs):
        super().__init__(args)

        self.cohort = load_cohort(args.task_id)
        self.partitions = partition_dataset_classes(
            self.cohort['training'],
            classes=pd.DataFrame.from_records(self.cohort['training'])['modality'],
            num_partitions=args.num_folds,
            shuffle=True,
            seed=args.seed,
        )

    # for skipping predicted subjects
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

    def loader_transform(self, *, on_predict: bool) -> monai.transforms.Compose:
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

    def normalize_transform(self, *, on_predict: bool) -> monai.transforms.Compose:
        all_keys = [self.args.img_key]
        spacing_modes = [GridSampleMode.BILINEAR]
        if not on_predict:
            all_keys.append(self.args.seg_key)
            spacing_modes.append(GridSampleMode.NEAREST)
        transforms = [monai.transforms.SpacingD(all_keys, pixdim=self.args.spacing, mode=spacing_modes)]
        if self.args.norm_intensity:
            transforms.extend([
                monai.transforms.NormalizeIntensityD(self.args.img_key),
                monai.transforms.ThresholdIntensityD(self.args.img_key, threshold=-5, above=True, cval=-5),
                monai.transforms.ThresholdIntensityD(self.args.img_key, threshold=5, above=False, cval=5),
                monai.transforms.ScaleIntensityD(self.args.img_key, minv=0, maxv=1),
            ])
        else:
            transforms.append(monai.transforms.ScaleIntensityRanged(
                self.args.img_key,
                a_min=self.args.a_min,
                a_max=self.args.a_max,
                b_min=0,
                b_max=1,
                clip=True,
            ))
        return monai.transforms.Compose(transforms)

    @property
    def aug_transform(self) -> monai.transforms.Compose:
        crop_transform = {
            'cls': monai.transforms.RandCropByLabelClassesD(
                [self.args.img_key, self.args.seg_key],
                label_key=self.args.seg_key,
                spatial_size=self.args.sample_shape,
                num_classes=self.args.num_seg_classes,
                num_samples=self.args.num_crop_samples,
            ),
            'pn': monai.transforms.RandCropByPosNegLabeld(
                keys=[self.args.img_key, self.args.seg_key],
                label_key=self.args.seg_key,
                spatial_size=self.args.sample_shape,
                pos=1,
                neg=1,
                num_samples=self.args.num_crop_samples,
                image_key=self.args.img_key,
                image_threshold=0,
            )
        }[self.args.crop]

        return monai.transforms.Compose([
            monai.transforms.SpatialPadD(
                [self.args.img_key, self.args.seg_key],
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT
            ),
            crop_transform,
            monai.transforms.RandFlipD([self.args.img_key, self.args.seg_key], prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD([self.args.img_key, self.args.seg_key], prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD([self.args.img_key, self.args.seg_key], prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.RandRotate90D([self.args.img_key, self.args.seg_key], prob=self.args.rotate_p, max_k=3),
            monai.transforms.RandScaleIntensityD(self.args.img_key, factors=0.1, prob=self.args.scale_p),
            monai.transforms.RandShiftIntensityD(self.args.img_key, offsets=0.1, prob=self.args.shift_p),
        ])

    @property
    def train_transform(self) -> Callable:
        # MONAI thinks Compose is randomized and skip cache, so we have to expand it lol
        return monai.transforms.Compose([
            *self.loader_transform(on_predict=False).transforms,
            *self.normalize_transform(on_predict=False).transforms,
            *self.aug_transform.transforms,
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
            *self.loader_transform(on_predict=False).transforms,
            *self.normalize_transform(on_predict=False).transforms,
        ])

    @property
    def predict_transform(self):
        return monai.transforms.Compose([
            self.loader_transform(on_predict=True),
            self.normalize_transform(on_predict=True),
        ])

class AmosSwinMAEDataModule(pl.LightningDataModule):
    def __init__(self, args: AmosArgs | SwinMAEArgs):
        super().__init__()
        self.args = args
        self.train_images, self.val_images = train_test_split([
            {args.img_key: subject[args.img_key]}
            for cohort in load_cohort(args.task_id).values()
            for subject in cohort
        ], test_size=args.val_size, random_state=args.seed)

    @property
    def train_transform(self):
        img_key = self.args.img_key
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(img_key),
            monai.transforms.AddChannelD(img_key),
            monai.transforms.OrientationD(img_key, axcodes='RAS'),
            monai.transforms.SpacingD(img_key, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.ScaleIntensityRangeD(
                img_key,
                a_min=self.args.a_min,
                a_max=self.args.a_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai.transforms.CropForegroundD(img_key, source_key=img_key),
            *(
                (monai.transforms.NormalizeIntensityD(img_key), )
                if self.args.norm_intensity else ()
            ),
            monai.transforms.SpatialPadD(
                img_key,
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT
            ),
            monai.transforms.RandSpatialCropSamplesD(
                img_key,
                roi_size=self.args.sample_shape,
                num_samples=self.args.num_crop_samples,
                random_size=False,
                random_center=True,
            ),
            monai.transforms.RandFlipD(img_key, prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(img_key, prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(img_key, prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.RandRotate90D(img_key, prob=self.args.rotate_p, max_k=3),
            monai.transforms.RandScaleIntensityD(img_key, factors=0.1, prob=self.args.scale_p),
            monai.transforms.RandShiftIntensityD(img_key, offsets=0.1, prob=self.args.shift_p),
            monai.transforms.Lambda(lambda data: data[img_key]),
        ])

    @property
    def val_transform(self):
        img_key = self.args.img_key
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(img_key),
            monai.transforms.AddChannelD(img_key),
            monai.transforms.OrientationD(img_key, axcodes='RAS'),
            monai.transforms.SpacingD(img_key, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.ScaleIntensityRangeD(
                img_key,
                a_min=self.args.a_min,
                a_max=self.args.a_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai.transforms.CropForegroundD(img_key, source_key=img_key),
            *(
                (monai.transforms.NormalizeIntensityD(img_key), )
                if self.args.norm_intensity else ()
            ),
            monai.transforms.SpatialPadD(
                img_key,
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT
            ),
            monai.transforms.RandSpatialCropSamplesD(
                img_key,
                roi_size=self.args.sample_shape,
                num_samples=self.args.num_crop_samples,
                random_size=False,
                random_center=True,
            ),
            # monai.transforms.RandFlipD(img_key, prob=self.args.flip_p, spatial_axis=0),
            # monai.transforms.RandFlipD(img_key, prob=self.args.flip_p, spatial_axis=1),
            # monai.transforms.RandFlipD(img_key, prob=self.args.flip_p, spatial_axis=2),
            # monai.transforms.RandRotate90D(img_key, prob=self.args.rotate_p, max_k=3),
            # monai.transforms.RandScaleIntensityD(img_key, factors=0.1, prob=self.args.scale_p),
            # monai.transforms.RandShiftIntensityD(img_key, offsets=0.1, prob=self.args.shift_p),
            monai.transforms.Lambda(lambda data: data[img_key]),
        ])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=CacheDataset(
                self.train_images,
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

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=CacheDataset(
                self.val_images,
                transform=self.val_transform,
                cache_num=self.args.val_cache_num,
                num_workers=self.args.dataloader_num_workers,
            ),
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )
