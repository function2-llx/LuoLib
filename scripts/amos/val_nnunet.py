import json
from pathlib import Path

from tqdm import tqdm

import monai
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
from nnunet.paths import network_training_output_dir

network_training_output_dir = Path(network_training_output_dir)

from umei.datasets.amos.datamodule import DATA_DIR

task_id = 2
fold_id = 0

output_dir = Path(network_training_output_dir) / '3d_fullres/Task666_AMOS/nnUNetTrainerV2__nnUNetPlansv2.1'
dataset_json = json.loads((Path(DATA_DIR) / f'task{task_id}_dataset-fold{fold_id}.json').read_text())

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(['pred', 'ref']),
    monai.transforms.AddChannelD(['pred', 'ref']),
    monai.transforms.AsDiscreteD(['pred', 'ref'], to_onehot=16),
    monai.transforms.ToTensorD(['pred', 'ref']),
])

dice_metric = DiceMetric(reduction=MetricReduction.MEAN_BATCH)

def main():
    for case in tqdm(dataset_json['validation'], ncols=80):
        subject = Path(case['image']).name
        x = list(output_dir.glob(f'fold_*/validation_raw/{subject}'))
        assert len(x) == 1
        data = loader({
            'pred': x[0],
            'ref': DATA_DIR / case['label']
        })
        m = dice_metric([data['pred']], [data['ref']])[0]
        print(m)
    m = dice_metric.aggregate()
    print(m)
    print(m[1:].mean().item())

if __name__ == '__main__':
    main()
