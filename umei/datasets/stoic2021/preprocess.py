from pathlib import Path

import numpy as np
import pandas as pd
import monai
from monai.data import Dataset, DatasetSummary
from monai.transforms import ResizeWithPadOrCrop, NormalizeIntensityD
import torch
# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')

data_dir = Path('cropped')
MIN_HU = -1024

def get_summary(transform: monai.transforms.Transform):
    ref = pd.read_csv('reference.csv')
    dataset = Dataset([{
        'img': data_dir / f'{patient}.nii.gz',
        'mask': data_dir / f'{patient}_mask.nii.gz',
    } for patient in ref['PatientID']], transform)
    return DatasetSummary(dataset, image_key='img', label_key='mask', num_workers=0)

def cal_target_spacing() -> tuple[float, float, float]:
    summary = get_summary(monai.transforms.Compose([
        monai.transforms.LoadImageD('img'),
        monai.transforms.AddChannelD('img'),
        monai.transforms.OrientationD('img', 'RAS'),
    ]))
    return summary.get_target_spacing()

def cal_stat(spacing: tuple[float, float, float]) -> tuple[float, float, np.ndarray]:
    summary = get_summary(monai.transforms.Compose([
        monai.transforms.LoadImageD(['img', 'mask']),
        monai.transforms.AddChannelD(['img', 'mask']),
        monai.transforms.OrientationD(['img', 'mask'], 'RAS'),
        monai.transforms.SpacingD(['img', 'mask'], pixdim=spacing, mode=["bilinear", "nearest"]),
    ]))
    summary.calculate_statistics(foreground_threshold=MIN_HU)
    return summary.data_mean, summary.data_std, summary.max_size

def preprocess(spacing: tuple[float, float, float], max_size: np.ndarray):
    monai.transforms.Compose([
        monai.transforms.LoadImageD('img'),
        monai.transforms.AddChannelD('img'),
        monai.transforms.OrientationD('img', 'RAS'),
        monai.transforms.SpacingD('img', pixdim=spacing),
        monai.transforms.NormalizeIntensityD('img', )
    ])

def main():
    loader = monai.transforms.Compose([
        monai.transforms.LoadImageD(['img', 'mask']),
        monai.transforms.AddChannelD(['img', 'mask']),
        monai.transforms.OrientationD(['img', 'mask'], 'RAS'),
        monai.transforms.SpacingD(['img', 'mask'], pixdim=(0.734375, 0.734375, 0.8), mode=["bilinear", "nearest"]),
    ])
    data = loader({
        'img': 'cropped/5503.nii.gz',
        'mask': 'cropped/5503_mask.nii.gz',
    })

    saver = monai.transforms.SaveImageD(
        ['img', 'mask'],
        resample=False,
        output_dir='save-test',
        data_root_dir='cropped',
        separate_folder=False,
        output_postfix='',
    )
    saver(data)

    # target_sp = cal_target_spacing()
    # target_sp = (0.734375, 0.734375, 0.8)
    # print('target spacing:', target_sp)
    # data_mean, data_std, max_size = cal_stat(target_sp)
    # print('intensity:', data_mean, data_std)
    # print('max size:', max_size)

if __name__ == '__main__':
    main()
