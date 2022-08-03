from __future__ import annotations

from pathlib import Path

from umei.utils import DataKey, DataSplit

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
            item[DataKey.SEG] = DATA_DIR / split_folder_map[DataSplit.TRAIN] / 'label' / item[DataKey.IMG].name.replace('img', 'label')
    return ret

