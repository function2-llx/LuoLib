from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

import monai

origin_dir = Path('origin')
img_dir = origin_dir / 'data' / 'mha'
mask_dir = Path('lungmask')
output_dir = Path('cropped')
output_dir.mkdir(exist_ok=True, parents=True)

MIN_HU = -1024

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(['img', 'mask']),
    monai.transforms.AddChannelD(['img', 'mask']),
    monai.transforms.OrientationD(['img', 'mask'], axcodes='RAS'),
    monai.transforms.ThresholdIntensityD('img', threshold=MIN_HU, above=True, cval=MIN_HU),
    monai.transforms.LambdaD('img', lambda x: x - MIN_HU),
    monai.transforms.MaskIntensityD('img', mask_key='mask'),
    monai.transforms.LambdaD('img', lambda x: x + MIN_HU),
])
cropper = monai.transforms.CropForegroundD(['img', 'mask'], source_key='mask')
saver = monai.transforms.Compose([
    monai.transforms.SaveImageD(
        'img',
        output_dir=output_dir,
        output_postfix='',
        resample=False,
        data_root_dir=str(img_dir),
        separate_folder=False,
        print_log=False,
    ),
    monai.transforms.SaveImageD(
        'mask',
        output_dir=output_dir,
        output_postfix='mask',
        resample=False,
        data_root_dir=str(mask_dir),
        separate_folder=False,
        print_log=False,
    ),
])

def process_subject(patient_id):
    data = loader({
        'img': img_dir / f'{patient_id}.mha',
        'mask': mask_dir / f'{patient_id}.nii.gz',
    })

    ret = {
        'w-o': data['img'].shape[1],
        'h-o': data['img'].shape[2],
        'd-o': data['img'].shape[3],
    }
    data = cropper(data)
    ret.update({
        'w-c': data['img'].shape[1],
        'h-c': data['img'].shape[2],
        'd-c': data['img'].shape[3],
        'sw': data['img_meta_dict']['spacing'][0],
        'sh': data['img_meta_dict']['spacing'][1],
        'sd': data['img_meta_dict']['spacing'][2],
    })
    _data = saver(data)
    return ret

def main():
    ref = pd.read_csv(origin_dir / 'metadata' / 'reference.csv')
    pd.DataFrame.from_records(
        process_map(process_subject, ref['PatientID'], max_workers=32, chunksize=1)
    ).to_excel('crop.xlsx', index=False)

if __name__ == '__main__':
    main()
