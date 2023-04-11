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
        'img': str(data_dir / f'{patient}.nii.gz'),
        'mask': str(data_dir / f'{patient}_mask.nii.gz'),
    } for patient in ref['PatientID']], transform)
    return DatasetSummary(dataset, image_key='img', label_key='mask', num_workers=32)

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
    summary.calculate_statistics()
    return summary.data_mean, summary.data_std, summary.max_size

def preprocess(spacing: tuple[float, float, float], max_size: np.ndarray):
    monai.transforms.Compose([
        monai.transforms.LoadImageD('img'),
        monai.transforms.AddChannelD('img'),
        monai.transforms.OrientationD('img', 'RAS'),
        monai.transforms.SpacingD('img', pixdim=spacing),
        monai.transforms.NormalizeIntensityD('img', )
    ])

loader: monai.transforms.Transform
saver = monai.transforms.SaveImageD(
    ['img', 'mask'],
    output_dir='resampled',
    output_postfix='',
    resample=False,
    data_root_dir=str(data_dir),
    separate_folder=False,
)

def process(patient_id):
    data = loader({
        'img': data_dir / f'{patient_id}.nii.gz',
        'mask': data_dir / f'{patient_id}_mask.nii.gz',
    })
    saver(data)

def main():
    global loader
    # target_sp = cal_target_spacing()
    target_sp = (0.734375, 0.734375, 0.8)
    print('target spacing:', target_sp)
    loader = monai.transforms.Compose([
        monai.transforms.LoadImageD(['img', 'mask']),
        monai.transforms.AddChannelD(['img', 'mask']),
        monai.transforms.OrientationD(['img', 'mask'], 'RAS'),
        monai.transforms.SpacingD(['img', 'mask'], pixdim=target_sp),
    ])
    process('92')
    process('1967')
    process('4888')
    # data_mean, data_std, max_size = cal_stat(target_sp)
    # data_mean, data_std, max_size = -729.2536010742188, 230.9741668701172, np.array([473, 377, 530])
    # print('max size:', max_size)

if __name__ == '__main__':
    main()
