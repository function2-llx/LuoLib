from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from umei.datamodule import SegDataModule, load_decathlon_datalist
from umei.utils import DataKey, DataSplit
from .omega import BTCVExpConf

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
    conf: BTCVExpConf

    def __init__(self, conf: BTCVExpConf, fold_id: int):
        super().__init__(conf)
        splits: list[dict[str, list[dict]]] = json.loads((DATA_DIR / 'splits.json').read_bytes())
        self.data = {
            data_split: [
                {
                    k: DATA_DIR / path
                    for k, path in case.items()
                }
                for case in cases
            ]
            for data_split, cases in splits[fold_id].items()
        }
        # for data_split in [DataSplit.TRAIN, DataSplit.VAL]:
        #     for case in self.data[data_split]:

        self.data[DataSplit.TRAIN] = np.random.choice(
            self.data[DataSplit.TRAIN],
            int(conf.data_ratio * len(self.data[DataSplit.TRAIN])),
            replace=False,
        ).tolist()

    def train_data(self) -> Sequence:
        return self.data[DataSplit.TRAIN]

    def val_data(self) -> Sequence:
        return self.data[DataSplit.VAL]

    def test_data(self) -> Sequence:
        return self.data[DataSplit.VAL]
