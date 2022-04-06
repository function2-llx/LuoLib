from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import StratifiedKFold
from monai.apps.datasets import DecathlonDataset

from umei.utils.args import UMeIArgs

class Stoic2021DataModule(LightningDataModule):
    def __init__(self, args: UMeIArgs):
        super().__init__()
        self.args = args
        self.val_id = -1
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

        return [
            train_cohort.iloc[fold_indices, :].to_dict('records')
            for fold_id, (_, fold_indices) in enumerate(skf.split(train_cohort.index, train_cohort['cls']))
        ]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != 'fit':
            return
        self.val_id += 1

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass
