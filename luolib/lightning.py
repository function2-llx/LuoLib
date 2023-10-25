from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from lightning import LightningModule as LightningModuleBase, Trainer as TrainerBase
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCheckpointBase
from lightning.pytorch.cli import LightningCLI as LightningCLIBase
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from timm.scheduler.scheduler import Scheduler as TIMMScheduler

from luolib.optim.factory import NamedParamGroup, OptimizerCallable, infer_weight_decay_keys, normalize_param_groups
from luolib.scheduler.factory import LRScheduler, LRSchedulerConfig, LRSchedulerConfigWithCallable

class LightningModule(LightningModuleBase):
    trainer: Trainer

    def __init__(
        self,
        optimizer: OptimizerCallable | None = None,
        lr_scheduler: LRSchedulerConfigWithCallable | None = None,
    ):
        super().__init__()
        self.optimizer_callable = optimizer
        self.lr_scheduler_config_with_callable = lr_scheduler

    def grad_named_parameters(self):
        for pn, p in self.named_parameters():
            if p.requires_grad:
                yield pn, p

    def get_param_groups(self) -> list[NamedParamGroup]:
        # this function does not take "no weight decay" into account
        return [{'params': list(self.grad_named_parameters())}]

    def get_decay_keys(self) -> set[str]:
        return infer_weight_decay_keys(self)

    def configure_optimizers(self):
        param_groups = self.get_param_groups()
        normalized_param_groups = normalize_param_groups(param_groups, self.get_decay_keys())
        self.optimizer = self.optimizer_callable(normalized_param_groups)
        self.lr_scheduler_config: LRSchedulerConfig = self.lr_scheduler_config_with_callable
        self.lr_scheduler_config.scheduler = self.lr_scheduler_config_with_callable.scheduler(self.optimizer)
        return [self.optimizer], [self.lr_scheduler_config]

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.trainer.log_dir / 'model.txt').write_text(repr(self))

    def on_train_start(self) -> None:
        # https://github.com/Lightning-AI/lightning/issues/17972
        match scheduler := self.lr_scheduler_config.scheduler:
            case TIMMScheduler():
                scheduler.step_update(0)

    def lr_scheduler_step(self, scheduler: LRScheduler, metric=None):
        match scheduler:
            case TIMMScheduler():
                scheduler.step_update(self.global_step + 1, metric)
            case _:
                super().lr_scheduler_step(scheduler, metric)

class Trainer(TrainerBase):
    @cached_property
    def log_dir(self) -> Path:
        """
        You must call this on all processes. Failing to do so will cause your program to stall forever.
        """
        logger = self.logger
        assert isinstance(logger, WandbLogger)
        if self.is_global_zero:
            log_dir = Path(logger.save_dir) / logger.experiment.name / f'{Path(logger.experiment.dir).parent.name}'
        else:
            log_dir = None
        log_dir = self.strategy.broadcast(log_dir)
        return log_dir

class ModelCheckpoint(ModelCheckpointBase):
    def __resolve_ckpt_dir(self, trainer: Trainer):
        return trainer.log_dir / 'checkpoint'

class LightningCLI(LightningCLIBase):
    model: LightningModule

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            model_class=LightningModule,
            save_config_kwargs={'config_filename': 'conf.yaml'},
            trainer_class=Trainer,
            subclass_mode_model=True,
            auto_configure_optimizers=False,
            **kwargs,
        )

    def before_instantiate_classes(self):
        # wandb wants to use a directory already existing: https://github.com/wandb/wandb/issues/714#issuecomment-565870686
        save_dir = self.config[self.subcommand].trainer.logger.init_args.save_dir
        Path(save_dir).mkdir(exist_ok=True, parents=True)
