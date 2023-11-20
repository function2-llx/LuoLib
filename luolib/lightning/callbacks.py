from __future__ import annotations

from lightning.pytorch.callbacks import ModelCheckpoint as ModelCheckpointBase

from .trainer import Trainer

class ModelCheckpoint(ModelCheckpointBase):
    def __resolve_ckpt_dir(self, trainer: Trainer):
        return trainer.log_dir / 'checkpoint'
