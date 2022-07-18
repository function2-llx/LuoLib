from __future__ import annotations

import monai
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from monai.data import DataLoader, CacheDataset
from monai.utils import GridSampleMode, NumpyPadMode
from umei.snim import SnimArgs
from umei.utils import DataSplit

def build_pretrain_datasets(args: SnimArgs) -> dict[str, Dataset]:
    # current implement a placeholder for all BTCV data

    from umei.datasets.btcv import load_cohort
    all_data = load_cohort(img_only=True, merge=True)
    train_data, val_data = train_test_split(all_data, test_size=args.val_size, random_state=args.seed)

    # split naming: following MSD convention
    return {
        DataSplit.TRAIN: CacheDataset(
            train_data,
            transform=monai.transforms.Compose([
                monai.transforms.LoadImageD('img'),
                monai.transforms.AddChannelD('img'),
                monai.transforms.OrientationD('img', axcodes='RAS'),
                monai.transforms.SpacingD('img', pixdim=args.spacing, mode=GridSampleMode.BILINEAR),
                monai.transforms.ScaleIntensityRangeD(
                    'img',
                    a_min=-175,
                    a_max=250,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                monai.transforms.CropForegroundD('img', source_key='img'),
                monai.transforms.SpatialPadD(
                    'img',
                    spatial_size=args.sample_shape,
                    mode=NumpyPadMode.CONSTANT,
                ),
                monai.transforms.RandSpatialCropD(
                    'img',
                    roi_size=args.sample_shape,
                    random_center=True,
                    random_size=False,
                ),
                monai.transforms.RandFlipD('img', prob=args.flip_p, spatial_axis=0),
                monai.transforms.RandFlipD('img', prob=args.flip_p, spatial_axis=1),
                monai.transforms.RandFlipD('img', prob=args.flip_p, spatial_axis=2),
                monai.transforms.RandRotate90D('img', prob=args.rotate_p, max_k=3),
                monai.transforms.RandScaleIntensityD('img', factors=args.scale_factor, prob=args.scale_p),
                monai.transforms.RandShiftIntensityD('img', offsets=args.shift_offset, prob=args.shift_p),
                monai.transforms.Lambda(lambda data: data['img']),
            ]),
            cache_num=args.train_cache_num,
        ),
        DataSplit.VAL: CacheDataset(
            val_data,
            transform=monai.transforms.Compose([
                monai.transforms.LoadImageD('img'),
                monai.transforms.AddChannelD('img'),
                monai.transforms.OrientationD('img', axcodes='RAS'),
                monai.transforms.SpacingD('img', pixdim=args.spacing, mode=GridSampleMode.BILINEAR),
                monai.transforms.ScaleIntensityRangeD(
                    'img',
                    a_min=-175,
                    a_max=250,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                monai.transforms.CropForegroundD('img', source_key='img'),
                monai.transforms.SpatialPadD(
                    'img',
                    spatial_size=args.sample_shape,
                    mode=NumpyPadMode.CONSTANT,
                ),
                monai.transforms.RandSpatialCropD(
                    'img',
                    roi_size=args.sample_shape,
                    random_center=True,
                    random_size=False,
                ),
                # monai.transforms.RandFlipD('img', prob=args.flip_p, spatial_axis=0),
                # monai.transforms.RandFlipD('img', prob=args.flip_p, spatial_axis=1),
                # monai.transforms.RandFlipD('img', prob=args.flip_p, spatial_axis=2),
                # monai.transforms.RandRotate90D('img', prob=args.rotate_p, max_k=3),
                # monai.transforms.RandScaleIntensityD('img', factors=args.scale_factor, prob=args.scale_p),
                # monai.transforms.RandShiftIntensityD('img', offsets=args.shift_offset, prob=args.shift_p),
                monai.transforms.Lambda(lambda data: data['img']),
            ]),
        ),
    }

class SnimDataModule(pl.LightningDataModule):
    # dataset: maybe ConcatDataset of multiple monai datasets with respective transform
    def __init__(self, args: SnimArgs, datasets: dict[str, Dataset]):
        super().__init__()
        self.args = args
        self.datasets = datasets

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.datasets[DataSplit.TRAIN],
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.datasets[DataSplit.VAL],
            batch_size=1,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )
