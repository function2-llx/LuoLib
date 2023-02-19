import itertools
from time import monotonic_ns, monotonic
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

data_dir = Path('datasets/AMOS22/all/imagesVa')
output_root = Path('benchmark-io')

def process(case: str):
    origin_path = data_dir / f'{case}.nii.gz'
    base_path = output_root / case
    shutil.copy2(origin_path, base_path.with_suffix('.nii.gz'))
    x: nib.Nifti1Image | nib.Nifti2Image = nib.load(origin_path)
    nib.save(x, base_path.with_suffix('.nii'))
    data = np.asanyarray(x.dataobj, dtype=x.get_data_dtype())
    np.save(str(base_path.with_suffix('.npy')), data)
    np.save(str(base_path.with_suffix('.npy').with_stem(f'{case}-affine')), x.affine)
    np.savez(base_path.with_suffix('.npz'), data=data, affine=x.affine)

def process_npz(case):
    x: nib.Nifti1Image = nib.load(output_root / f'{case}.nii')
    np.savez((output_root / f'{case}.npz').with_suffix('.npz'), data=x.dataobj, affine=x.affine)

def read(case: str, suffix: str):
    path = output_root / f'{case}{suffix}'
    match suffix:
        case '.nii' | '.nii.gz':
            x = nib.load(path)
            x = np.array(x.dataobj)
        case '.npy':
            x = np.load(str(path)).astype(np.float64)
            affine = np.load(str(path.with_stem(f'{case}-affine')))
        case '.npz':
            x = np.load(str(path))
            data = x['data'].astype(np.float64)
            affine = x['affine']

def main():
    cases = [path.with_suffix('').stem for path in data_dir.iterdir()]
    if not output_root.exists():
        output_root.mkdir()
        process_map(process, cases, dynamic_ncols=True, max_workers=8)
    test_suffixes = ['.npy', '.npz', '.nii', '.nii.gz']
    for suffix in test_suffixes:
        start = monotonic()
        process_map(read, cases, itertools.repeat(suffix), dynamic_ncols=80, desc=suffix, max_workers=8)
        # for case in tqdm(cases, desc=suffix, dynamic_ncols=True):
        #     read(case, suffix)
        print(f'{suffix} elapsed {(monotonic() - start):.3f}s')

if __name__ == '__main__':
    main()
