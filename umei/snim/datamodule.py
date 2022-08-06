from __future__ import annotations

from collections.abc import Sequence

import monai
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from monai.data import DataLoader, CacheDataset
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datamodule import UMeIDataModule
from umei.snim import SnimArgs
from umei.utils import DataKey, DataSplit

def build_pretrain_data(args: SnimArgs) -> dict[str, Sequence]:
    # current implement a placeholder for all AMOS and BTCV data
    from umei.datasets.btcv import load_cohort as btcv_load
    btcv_data = btcv_load(img_only=True, merge=True)
    btcv_train_data, btcv_val_data = train_test_split(btcv_data, test_size=args.val_size, random_state=args.seed)

    from umei.datasets.amos import load_cohort as amos_load
    amos_data = amos_load(task_id=1, merge=True)
    amos_train_data, amos_val_data = train_test_split(amos_data, test_size=args.val_size, random_state=args.seed)

    return {
        DataSplit.TRAIN: btcv_train_data + amos_train_data,
        DataSplit.VAL: amos_val_data + btcv_val_data,
    }

class SnimDataModule(UMeIDataModule):
    args: SnimArgs

    # dataset: maybe ConcatDataset of multiple monai datasets with respective transform
    def __init__(self, args: SnimArgs, data: dict[str, Sequence]):
        super().__init__(args)
        self.data = data

    @UMeIDataModule.train_transform.getter
    def train_transform(self):
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(DataKey.IMG),
            monai.transforms.AddChannelD(DataKey.IMG),
            monai.transforms.OrientationD(DataKey.IMG, axcodes='RAS'),
            monai.transforms.SpacingD(DataKey.IMG, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.ScaleIntensityRangeD(
                DataKey.IMG,
                a_min=-175,
                a_max=250,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai.transforms.CropForegroundD(DataKey.IMG, source_key=DataKey.IMG),
            monai.transforms.SpatialPadD(
                DataKey.IMG,
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT,
            ),
            monai.transforms.RandSpatialCropD(
                DataKey.IMG,
                roi_size=self.args.sample_shape,
                random_center=True,
                random_size=False,
            ),
            monai.transforms.RandFlipD(DataKey.IMG, prob=self.args.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(DataKey.IMG, prob=self.args.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(DataKey.IMG, prob=self.args.flip_p, spatial_axis=2),
            monai.transforms.RandRotate90D(DataKey.IMG, prob=self.args.rotate_p, max_k=3),
            monai.transforms.RandScaleIntensityD(DataKey.IMG, factors=self.args.scale_factor, prob=self.args.scale_p),
            monai.transforms.RandShiftIntensityD(DataKey.IMG, offsets=self.args.shift_offset, prob=self.args.shift_p),
            monai.transforms.Lambda(lambda data: data[DataKey.IMG]),
        ])

    @UMeIDataModule.val_transform.getter
    def val_transform(self):
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(DataKey.IMG),
            monai.transforms.AddChannelD(DataKey.IMG),
            monai.transforms.OrientationD(DataKey.IMG, axcodes='RAS'),
            monai.transforms.SpacingD(DataKey.IMG, pixdim=self.args.spacing, mode=GridSampleMode.BILINEAR),
            monai.transforms.ScaleIntensityRangeD(
                DataKey.IMG,
                a_min=-175,
                a_max=250,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai.transforms.CropForegroundD(DataKey.IMG, source_key=DataKey.IMG),
            monai.transforms.SpatialPadD(
                DataKey.IMG,
                spatial_size=self.args.sample_shape,
                mode=NumpyPadMode.CONSTANT,
            ),
            monai.transforms.RandSpatialCropD(
                DataKey.IMG,
                roi_size=self.args.sample_shape,
                random_center=True,
                random_size=False,
            ),
            monai.transforms.Lambda(lambda data: data[DataKey.IMG]),
        ])

    def train_data(self) -> Sequence:
        return self.data[DataSplit.TRAIN]

    def val_data(self) -> Sequence:
        return self.data[DataSplit.VAL]
