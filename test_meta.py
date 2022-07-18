from pathlib import Path
import tempfile

from einops import rearrange
import nibabel as nib
import numpy as np

import monai
from monai.data import DataLoader, Dataset

def main():
    with tempfile.TemporaryDirectory() as tempdir:
        img_path = Path(tempdir) / 'test.nii.gz'
        nib.save(nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4)), img_path)
        data = next(iter(DataLoader(
            Dataset(
                [{'img': img_path}],
                monai.transforms.Compose([
                    monai.transforms.LoadImageD('img'),
                    monai.transforms.AddChannelD('img'),
                ])
            ),
            batch_size=1,
        )))
        x = data['img'].as_tensor()
        x = rearrange(x.repeat(1, 2, 1, 1, 1), 'n c h w d -> c n h w d')
        print(x.shape)
        z = x[1]

if __name__ == '__main__':
    main()
