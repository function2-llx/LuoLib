from itertools import repeat
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

import monai.transforms
from nnunet.paths import nnUNet_raw_data as nnunet_rd_dir, preprocessing_output_dir as nnunet_pp_output_dir
from umei.datasets.amos import AmosArgs
from umei.datasets.amos.datamodule import DATASET_ROOT, load_cohort
from umei.utils import UMeIParser

task_name = 'Task666_AMOS'

nnunet_rd_dir = Path(nnunet_rd_dir) / task_name
nnunet_pp_output_dir = Path(nnunet_pp_output_dir) / task_name

def process(case: dict, args: AmosArgs, loader: Callable[[dict], dict]) -> dict:
    data = loader(case)
    shape = data[args.img_key].shape
    return {
        'subject': case['subject'],
        'w': shape[1],
        'h': shape[2],
        'd': shape[3],
    }

def main():
    parser = UMeIParser((AmosArgs, ), use_conf=True)
    args: AmosArgs = parser.parse_args_into_dataclasses()[0]
    cohort = load_cohort(args)['training']
    properties = pd.read_pickle(nnunet_pp_output_dir / 'dataset_properties.pkl')
    all_spacings = properties['all_spacings']
    spacing = np.median(all_spacings, axis=0)
    spacing = np.array([spacing[1], spacing[2], spacing[0]])
    print('median spacing =', spacing)
    loader = monai.transforms.Compose([
        monai.transforms.LoadImageD(args.img_key),
        monai.transforms.AddChannelD(args.img_key),
        monai.transforms.OrientationD(args.img_key, 'RAS'),
        monai.transforms.SpacingD(args.img_key, pixdim=spacing),
    ])
    pd.DataFrame.from_records(
        process_map(process, cohort, repeat(args), repeat(loader), ncols=80, max_workers=16)
    ).set_index('subject').to_excel(DATASET_ROOT / 'stat.xlsx')

if __name__ == '__main__':
    main()
