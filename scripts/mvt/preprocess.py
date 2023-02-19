import itertools as it
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

import monai
from monai import transforms as monai_t
from monai.data import MetaTensor

output_root = Path('mvt-data')

def process(task: tuple[Path, Path]):
    img_path, output_dir = task
    if output_dir.exists():
        return
    output_dir.mkdir(parents=True)
    loader = monai.transforms.Compose([
        monai_t.LoadImage(image_only=True, ensure_channel_first=True),
        monai_t.CropForeground('min'),
    ])
    img: MetaTensor = loader(img_path)

    np.save(str(output_dir / 'data.npy'), img.numpy())
    np.save(str(output_dir / 'affine.npy'), img.affine)

def main():
    global output_root

    tasks = []
    for splits, splits_name in [
        (['Tr', 'Va'], 'train+val'),
        (['Ts'], 'test'),
    ]:
        output_dir = output_root / 'AMOS22' / splits_name
        # this is the only one that PyCharm could recognize type (it.chain, itz.concat not work)
        for img_path in it.chain.from_iterable([
            (Path('datasets/AMOS22/all') / f'images{split}').glob('*.nii.gz')
            for split in splits
        ]):
            case = img_path.with_suffix('').stem
            case_id = int(case[-4:])
            modality = 'CT' if case_id <= 500 else 'MRI'
            tasks.append((img_path, output_dir / modality / case))

    process_map(process, tasks, max_workers=8, dynamic_ncols=True, desc='AMOS22')

if __name__ == '__main__':
    main()
