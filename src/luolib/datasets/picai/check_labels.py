from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map

from monai.transforms import LoadImageD

loader = LoadImageD('seg')

data_dir = Path('labels/csPCa_lesion_delineations/human_expert/resampled')

def process(filepath: Path):
    data = loader({'seg': filepath})
    seg = data['seg']
    return {
        'file': filepath.name,
        'sum': seg.sum(),
        'ratio': seg.sum() / seg.size,
        'values': ' '.join(map(str, np.unique(seg))),
    }

def main():
    pd.DataFrame.from_records(
        process_map(process, list(data_dir.iterdir()), ncols=80, max_workers=15)
    ).set_index('file').to_excel('labels.xlsx')

if __name__ == '__main__':
    main()
