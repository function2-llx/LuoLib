from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# from monai.data import load_decathlon_datalist
from umei.datamodule import SegDataModule, load_decathlon_datalist
from umei.utils import DataKey, DataSplit
from .args import BTCVArgs

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

split_folder_map = {
    DataSplit.TRAIN: 'Training',
    DataSplit.TEST: 'Testing',
}

def load_cohort(img_only: bool = False, merge: bool = False):
    if merge:
        assert img_only
        return [
            {DataKey.IMG: img_path}
            for img_path in DATA_DIR.glob('*/img/*.nii.gz')
        ]
    ret = {
        split: [
            {DataKey.IMG: filepath}
            for filepath in (DATA_DIR / folder / 'img').glob('*.nii.gz')
        ]
        for split, folder in split_folder_map.items()
    }
    if not img_only:
        for item in ret[DataSplit.TRAIN]:
            item[DataKey.SEG] = DATA_DIR / split_folder_map[DataSplit.TRAIN] / \
                                'label' / item[DataKey.IMG].name.replace('img', 'label')
    return ret

class BTCVDataModule(SegDataModule):
    args: BTCVArgs

    def __init__(self, args: BTCVArgs):
        super().__init__(args)
        self.data = {
            split: load_decathlon_datalist(
                DATA_DIR / 'dataset_0.json',
                data_list_key=key,
            )
            for split, key in [
                (DataSplit.TRAIN, 'training'),
                (DataSplit.VAL, 'validation'),
            ]
        }

    def train_data(self) -> Sequence:
        return self.data[DataSplit.TRAIN]

    def val_data(self) -> dict[DataSplit, Sequence]:
        return {
            DataSplit.VAL: self.data[DataSplit.VAL]
        }
