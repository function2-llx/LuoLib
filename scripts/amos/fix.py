from pathlib import Path

import numpy as np

import monai
import nibabel as nib

fix_ids = ['amos_0207', 'amos_0244', 'amos_0247', 'amos_0287']

fix_transform = monai.transforms.Compose([
    monai.transforms.LoadImageD('img'),
    monai.transforms.AddChannelD('img'),
    monai.transforms.FlipD('img', spatial_axis=0)
])

def main():
    for fix_id in fix_ids:
        file_path = Path('../../amos-final-pred') / f'{fix_id}.nii.gz'
        origin_nib = nib.load(file_path)
        fixed_img = fix_transform({'img': file_path})['img']
        nib.save(
            nib.Nifti1Image(
                fixed_img[0].astype(np.uint8),
                affine=origin_nib.affine,
                header=origin_nib.header,
            ),
            file_path.with_name(f'{fix_id}-fix.nii.gz'),
        )

if __name__ == '__main__':
    main()
