from typing import Optional

from pytorch_lightning import LightningDataModule

from umei.utils import UMeIArgs

class CVDataModule(LightningDataModule):
    def __init__(self, args: UMeIArgs):
        super().__init__()
        self.args = args
        self.val_id = 0

    @property
    def val_id(self) -> int:
        return self._val_id

    @val_id.setter
    def val_id(self, x: int):
        assert x in range(self.num_cv_folds)
        self._val_id = x

    @property
    def num_cv_folds(self) -> int:
        return self.args.num_folds - self.args.use_test_fold

    @property
    def val_parts(self) -> dict[str, int]:
        ret = {'val': self.val_id}
        if self.args.use_test_fold:
            ret['test'] = self.args.num_folds - 1
        return ret
