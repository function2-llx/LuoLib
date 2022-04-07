from typing import Optional

from pytorch_lightning import LightningDataModule

class CVDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self._val_id = -1

    @property
    def val_id(self) -> int:
        return self._val_id

    @val_id.setter
    def val_id(self, x: int):
        self._val_id = x
