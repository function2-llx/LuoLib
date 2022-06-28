from copy import deepcopy
import json
from pathlib import Path

from umei.datasets.amos import AmosArgs, AmosDataModule
from umei.datasets.amos.datamodule import DATA_DIR

task_id = 1

def main():
    args = AmosArgs(task_id=task_id, num_folds=5, seed=42)
    args.task_id = 1
    datamodule = AmosDataModule(args)
    dataset_json = json.loads((DATA_DIR / f'task{task_id}_dataset.json').read_text())
    print(dataset_json)
    for val_fold_id in range(args.num_folds):
        fold_dataset_json = deepcopy(dataset_json)
        fold_dataset_json['training'] = [
            {
                'image': str(case['img'].relative_to(DATA_DIR)),
                'label': str(case['seg'].relative_to(DATA_DIR)),
            }
            for i in range(args.num_folds) if i != val_fold_id
            for case in datamodule.partitions[i]
        ]
        fold_dataset_json['numTraining'] = len(fold_dataset_json['training'])
        fold_dataset_json['validation'] = [
            {
                'image': str(case['img'].relative_to(DATA_DIR)),
                'label': str(case['seg'].relative_to(DATA_DIR)),
            }
            for case in datamodule.partitions[val_fold_id]
        ]
        fold_dataset_json['numValidation'] = len(fold_dataset_json['validation'])
        Path(DATA_DIR / f'task{task_id}_dataset-fold{val_fold_id}.json').write_text(
            json.dumps(fold_dataset_json, indent=4, ensure_ascii=False)
        )

if __name__ == '__main__':
    main()
