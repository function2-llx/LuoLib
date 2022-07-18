from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import default_collate

import monai
from monai.data import DataLoader, Dataset, partition_dataset_classes
from monai.utils import GridSampleMode, NumpyPadMode
from umei.datamodule import CVDataModule

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort(img_only: bool = False, merge: bool = False):
    # placeholder to load all image data
    assert img_only and merge

    return [
        {'img': img_path}
        for img_path in DATA_DIR.glob('*/img/*.nii.gz')
    ]

