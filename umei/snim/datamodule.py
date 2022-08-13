from __future__ import annotations

from collections.abc import Sequence

from sklearn.model_selection import train_test_split

import monai
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datamodule import UMeIDataModule
from umei.snim import SnimArgs
from umei.utils import DataKey, DataSplit

def build_pretrain_data(args: SnimArgs) -> dict[str, Sequence]:
    train_data = []
    val_data = []

    for dataset in args.datasets:
        if dataset == 'btcv':
            from umei.datasets.btcv import load_cohort
            data = load_cohort(img_only=True, merge=True)
        elif dataset == 'amos':
            from umei.datasets.amos import load_cohort
            data = load_cohort(task_id=1, merge=True)
        elif dataset == 'act1k':
            from umei.datasets.act1k import load_cohort
            data = load_cohort()
        else:
            raise RuntimeError

        train_part, val_part = train_test_split(data, test_size=args.val_size, random_state=args.seed)
        train_data.extend(train_part)
        val_data.extend(val_part)

    return {
        DataSplit.TRAIN: train_data,
        DataSplit.VAL: val_data,
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
                a_min=self.args.a_min,
                a_max=self.args.a_max,
                b_min=self.args.b_min,
                b_max=self.args.b_max,
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
