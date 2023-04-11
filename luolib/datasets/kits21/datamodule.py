from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from umei.datamodule import CVDataModule, SegDataModule
from umei.utils import DataKey, DataSplit
from .args import KiTS21Args

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

class KiTS21DataModule(SegDataModule):
    args: KiTS21Args

    def __init__(self, args: KiTS21Args):
        super().__init__(args)
        data_info = pd.read_json(DATA_DIR / 'fold0.json')
        self.data = {
            split: [
                {
                    DataKey.IMG: DATA_DIR / case_id / 'imaging.nii.gz',
                    DataKey.SEG: DATA_DIR / case_id / 'aggregated_MAJ_seg.nii.gz',
                }
                for case_id in data_info[split_key]
            ]
            for split, split_key in [
                (DataSplit.TRAIN, 'train'),
                (DataSplit.VAL, 'val'),
            ]
        }

    def train_data(self) -> Sequence:
        return self.data[DataSplit.TRAIN]

    def val_data(self) -> dict[DataSplit, Sequence]:
        return {
            DataSplit.VAL: self.data[DataSplit.VAL]
        }

    def test_data(self) -> Sequence:
        return self.data[DataSplit.VAL]
