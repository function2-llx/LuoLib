from pathlib import Path

from umei.utils import DataKey

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort(image_only: bool = True):
    if not image_only:
        raise NotImplementedError
    return [
        {DataKey.IMG: DATA_DIR / 'Image' / f'Case_{case_id:05d}_0000.nii.gz'}
        for case_id in range(1, 1063)
    ]

ACT1K_DATA_DIR = DATA_DIR
