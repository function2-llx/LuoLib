
from collections.abc import Sequence
from pathlib import Path


from sklearn.model_selection import train_test_split

import monai
from monai import transforms as monai_t
from monai.utils import GridSampleMode, NumpyPadMode, PytorchPadMode

from umei.datamodule_omega import ExpDataModuleBase
from umei.snim.omega import SnimConf
from umei.transforms import RandAffineCropD, RandSpatialCenterGeneratorD, SimulateLowResolutionD, SpatialRangeGenerator
from umei.utils import DataKey, DataSplit

DATA_DIR = Path('snim-data')

class SnimDataModule(ExpDataModuleBase):
    conf: SnimConf

    # dataset: maybe ConcatDataset of multiple monai datasets with respective transform
    def __init__(self, conf: SnimConf):
        super().__init__(conf)

        train_data = []
        val_data = []

        for dataset in conf.datasets:
            data = [
                {DataKey.IMG: path}
                for path in (DATA_DIR / dataset).glob('*.npy')
            ]
            train_part, val_part = train_test_split(data, test_size=conf.val_size, random_state=conf.seed)
            train_data.extend(train_part)
            val_data.extend(val_part)

        self.data = {
            DataSplit.TRAIN: train_data,
            DataSplit.VAL: val_data,
        }

    def train_transform(self):
        conf = self.conf
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(DataKey.IMG, image_only=True, ensure_channel_first=True),
            monai_t.SpatialPadD(
                DataKey.IMG,
                spatial_size=conf.sample_shape,
                mode=PytorchPadMode.CONSTANT,
                pad_min=True,
            ),
            RandAffineCropD(
                DataKey.IMG,
                conf.sample_shape,
                GridSampleMode.BILINEAR,
                dummy_dim=conf.dummy_dim,
                center_generator=RandSpatialCenterGeneratorD(DataKey.IMG, conf.sample_shape),
                rotate_generator=SpatialRangeGenerator(
                    conf.rotate_range,
                    conf.rotate_p,
                    repeat=3 if conf.dummy_dim is None else 1,
                ),
                scale_generator=SpatialRangeGenerator(
                    conf.scale_range,
                    conf.scale_p,
                    default=1.,
                    repeat=3 if conf.dummy_dim is None else 2,
                ),
            ),
            SimulateLowResolutionD(
                DataKey.IMG, conf.simulate_low_res_zoom_range, conf.simulate_low_res_p, conf.dummy_dim
            ),
            monai.transforms.RandFlipD(DataKey.IMG, prob=conf.flip_p, spatial_axis=0),
            monai.transforms.RandFlipD(DataKey.IMG, prob=conf.flip_p, spatial_axis=1),
            monai.transforms.RandFlipD(DataKey.IMG, prob=conf.flip_p, spatial_axis=2),
            monai_t.Lambda(lambda data: data[DataKey.IMG]),
        ])

    def val_transform(self):
        conf = self.conf
        return monai.transforms.Compose([
            monai.transforms.LoadImageD(DataKey.IMG, image_only=True, ensure_channel_first=True),
            monai_t.SpatialPadD(
                DataKey.IMG,
                spatial_size=conf.sample_shape,
                mode=NumpyPadMode.CONSTANT,
                pad_min=True,
            ),
            monai_t.RandSpatialCropD(
                DataKey.IMG,
                roi_size=conf.sample_shape,
                random_center=True,
                random_size=False,
            ),
            monai_t.Lambda(lambda data: data[DataKey.IMG]),
        ])

    def train_data(self) -> Sequence:
        return self.data[DataSplit.TRAIN]

    def val_data(self) -> Sequence:
        return self.data[DataSplit.VAL]
