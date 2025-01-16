import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from umei.utils import DataKey, DataSplit

def main():
    cases = np.array(sorted([
        filepath.with_suffix('').stem[3:]
        for filepath in Path('origin/Training/img').iterdir()
    ]))
    num_folds = 5
    groups = np.arange(0, 5).repeat(6)
    assert len(cases) % num_folds == 0
    splits = []
    for train, val in GroupKFold(num_folds).split(cases, groups=groups):
        splits.append({
            data_split: [
                {
                    DataKey.IMG: f'Training/img/img{case}.nii.gz',
                    DataKey.SEG: f'Training/label/label{case}.nii.gz'
                }
                for case in cases[index]
            ]
            for data_split, index in [
                (DataSplit.TRAIN, train),
                (DataSplit.VAL, val)
            ]
        })
    Path('origin/splits.json').write_text(json.dumps(splits, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
