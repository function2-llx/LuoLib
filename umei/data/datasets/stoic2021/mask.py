from pathlib import Path
import os

import SimpleITK as sitk
from lungmask import mask as lungmask
import pandas as pd
from tqdm import tqdm

origin_dir = Path('origin')
data_dir = origin_dir / 'data' / 'mha'
output_dir = Path('lungmask')

def main():
    from sys import argv
    l, r, batch_size = map(int, argv[1:4])
    os.environ['CUDA_VISIBLE_DEVICES'] = argv[4]
    print(l, r)
    ref = pd.read_csv(origin_dir / 'metadata' / 'reference.csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    for patient_id in tqdm(ref['PatientID'][l:r]):
        img = sitk.ReadImage(str(data_dir / f'{patient_id}.mha'))
        mask = lungmask.apply(img, batch_size=batch_size)
        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(output_dir / f'{patient_id}.nii.gz'))

if __name__ == '__main__':
    main()
