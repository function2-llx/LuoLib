from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
import os
from pathlib import Path
from typing import Literal
import warnings

from lightning import LightningDataModule, LightningModule as LightningModuleBase, Trainer as TrainerBase
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint as ModelCheckpointBase, ModelSummary
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI as LightningCLIBase
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from lightning.pytorch.utilities.types import STEP_OUTPUT
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
import torch
from torch.optim import Optimizer

from monai.utils import ensure_tuple

from luolib.utils.grad import grad_norm
from luolib.datamodule import ExpDataModuleBase, CrossValDataModule
from luolib.optim.factory import NamedParamGroup, OptimizerCallable, infer_weight_decay_keys, normalize_param_groups
from luolib.scheduler.factory import LRScheduler, LRSchedulerConfig, LRSchedulerConfigWithCallable
from luolib.utils import fall_back_none

__all__ = [
    'LightningModule',
    'Trainer',
    'ModelCheckpoint',
    'LightningCLI',
    'LightningCLICrossVal',
]

class LightningModule(LightningModuleBase):
    trainer: Trainer

    def __init__(self, *, log_grad_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.log_grad_norm = log_grad_norm

    def grad_named_parameters(self):
        for pn, p in self.named_parameters():
            if p.requires_grad:
                yield pn, p

    def get_param_groups(self) -> list[NamedParamGroup]:
        # this function does not take "no weight decay" into account, will be normalized later
        return [{'params': list(self.grad_named_parameters())}]

    def get_decay_keys(self) -> set[str]:
        return infer_weight_decay_keys(self)

    def set_cli_optimizers(
        self,
        cli_optimizers: Sequence[OptimizerCallable],
        cli_lr_scheduler_configs: Sequence[LRSchedulerConfigWithCallable],
    ):
        if len(cli_optimizers) != len(cli_lr_scheduler_configs):
            pass
        self.cli_optimizers = cli_optimizers
        self.cli_lr_scheduler_configs = cli_lr_scheduler_configs

    def configure_optimizers(self):
        param_groups = self.get_param_groups()
        normalized_param_groups = normalize_param_groups(param_groups, self.get_decay_keys())
        optimizer = self.trainer.optimizer_callable(normalized_param_groups)
        lr_scheduler_config = LRSchedulerConfig(**vars(self.trainer.lr_scheduler_config_with_callable))
        lr_scheduler_config.scheduler = self.trainer.lr_scheduler_config_with_callable.scheduler(optimizer)
        # vars(scheduler) due to: https://github.com/Lightning-AI/lightning/issues/18870
        return [optimizer], [vars(lr_scheduler_config)]

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.trainer.log_dir / 'model.txt').write_text(repr(self))

    def on_train_start(self) -> None:
        # handle warm-up: https://github.com/Lightning-AI/lightning/issues/17972
        for scheduler in ensure_tuple(self.lr_schedulers()):
            if isinstance(scheduler, TIMMScheduler):
                scheduler.step_update(0)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: ..., batch_idx: int) -> None:
        match outputs:
            case torch.Tensor():
                loss = outputs
            case dict():
                loss = outputs['loss']
            case None:
                return
            case _:
                raise ValueError

        if loss.isfinite() or (save_dir := self.trainer.log_dir / 'bad-loss-ctx').exists():
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(save_dir / 'checkpoint.ckpt')
        torch.save(batch, save_dir / 'batch.pt')

    def lr_scheduler_step(self, scheduler: LRScheduler, metric=None):
        match scheduler:
            case TIMMScheduler():
                scheduler.step_update(self.global_step + 1, metric)
            case _:
                super().lr_scheduler_step(scheduler, metric)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.log_grad_norm:
            self.log('grad_norm', grad_norm(self))

    @property
    def datamodule(self) -> LightningDataModule:
        return self.trainer.datamodule

class Trainer(TrainerBase):
    def __init__(
        self,
        *,
        precision: _PRECISION_INPUT = '16-mixed',
        check_val_every_n_epoch: int | None = None,
        benchmark: bool = True,
        optimizer: OptimizerCallable | None = None,
        lr_scheduler: LRSchedulerConfigWithCallable | None = None,
        **kwargs,
    ):
        super().__init__(
            precision=precision,
            check_val_every_n_epoch=check_val_every_n_epoch,
            benchmark=benchmark,
            **kwargs,
        )
        self.optimizer_callable = optimizer
        self.lr_scheduler_config_with_callable = lr_scheduler

        from monai.config import USE_COMPILED
        if not USE_COMPILED:
            warnings.warn('MONAI is not using compiled')

    @cached_property
    def log_dir(self) -> Path:
        """
        You must call this on all processes. Failing to do so will cause your program to stall forever.
        """
        if self.is_global_zero:
            logger = self.logger
            assert isinstance(logger, WandbLogger)
            seed = os.getenv('PL_GLOBAL_SEED')
            root_dir = Path(logger.save_dir)
            log_dir = root_dir / f'seed-{seed}' / f'{Path(logger.experiment.dir).parent.name}'
        else:
            log_dir = None
        log_dir = self.strategy.broadcast(log_dir)
        return log_dir

    def play(self, *, ckpt_path: str | None, **kwargs):
        """have to define this function in Trainer or CLI can't find the `ckpt_path` parameter"""

class ModelCheckpoint(ModelCheckpointBase):
    def __resolve_ckpt_dir(self, trainer: Trainer):
        return trainer.log_dir / 'checkpoint'

class LightningCLI(LightningCLIBase):
    _subcommand_preparing: str | None = None
    trainer: Trainer

    def __init__(
        self,
        *,
        model_class: type[LightningModule] | Callable[..., LightningModule] | None = LightningModule,
        datamodule_class: type[ExpDataModuleBase] | Callable[..., ExpDataModuleBase] | None = ExpDataModuleBase,
        save_config_kwargs: dict[str, ...] | None = None,
        trainer_class: type[Trainer] | Callable[..., Trainer] = Trainer,
        trainer_defaults: dict[str, ...] | None = None,
        seed_everything_default: bool | int = 42,
        parser_kwargs: dict[str, ...] | dict[str, dict[str, ...]] | None = None,
        subclass_mode_model: bool = True,
        subclass_mode_data: bool = True,
        auto_configure_optimizers: bool = False,
        **kwargs,
    ):
        save_config_kwargs = fall_back_none(save_config_kwargs, {'config_filename': 'conf.yaml'})
        trainer_defaults = fall_back_none(trainer_defaults, {
            'callbacks': [
                LearningRateMonitor(),
                ModelSummary(max_depth=2),

            ]
        })
        parser_kwargs = fall_back_none(parser_kwargs, {'parser_mode': "omegaconf"})
        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_kwargs=save_config_kwargs,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default,
            parser_kwargs=parser_kwargs,
            subclass_mode_model=subclass_mode_model,
            subclass_mode_data=subclass_mode_data,
            auto_configure_optimizers=auto_configure_optimizers,
            **kwargs,
        )

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        subcommands = LightningCLIBase.subcommands()
        return {
            **subcommands,
            'play': set(),
        }

    def _prepare_subcommand_parser(self, klass: type, subcommand: str, **kwargs) -> LightningArgumentParser:
        # make current subcommand available in `add_arguments_to_parser`
        self._subcommand_preparing = subcommand
        return super()._prepare_subcommand_parser(klass, subcommand, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--float32_matmul_precision', type=Literal['medium', 'high', 'highest'], default='high')
        parser.link_arguments('trainer.max_steps', 'data.init_args.dataloader.num_batches')
        parser.add_argument('--compile', type=bool, default=True)
        parser.add_argument('--trace_numpy', type=bool, default=False)
        parser.add_argument('--logger', type=WandbLogger)
        parser.link_arguments('logger', 'trainer.logger', apply_on='instantiate')
        parser.add_argument('--mp_start_method', type=Literal['fork', 'spawn', 'forkserver'], default='fork')

    def before_instantiate_classes(self):
        logger_args = self.config[self.subcommand].logger.init_args
        save_dir = Path(logger_args.save_dir) / logger_args.name
        # wandb wants to use a directory already existing: https://github.com/wandb/wandb/issues/714#issuecomment-565870686
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        logger_args.save_dir = str(save_dir)

    def _run_subcommand(self, subcommand: str):
        torch.multiprocessing.set_start_method(self._get(self.config, 'mp_start_method'))
        torch.set_float32_matmul_precision(self._get(self.config, 'float32_matmul_precision'))
        super()._run_subcommand(subcommand)

    def fit(self, model, **kwargs):
        # https://github.com/Lightning-AI/lightning/issues/17283
        if self._get(self.config, 'compile'):
            # https://github.com/pytorch/pytorch/issues/112335
            from torch._dynamo import config
            config.trace_numpy = self._get(self.config, 'trace_numpy')
            model = torch.compile(model)
        self.trainer.fit(model, **kwargs)

def fit_or_val(command: str):
    return command in ['fit', 'validate', 'play']

class LightningCLICrossVal(LightningCLI):
    datamodule: CrossValDataModule

    def __init__(
        self,
        *,
        datamodule_class: type[CrossValDataModule] | Callable[..., CrossValDataModule] | None = CrossValDataModule,
        **kwargs,
    ):
        super().__init__(datamodule_class=datamodule_class, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        if fit_or_val(self._subcommand_preparing):
            parser.add_argument('fold_id', type=int)
        super().add_arguments_to_parser(parser)

    def before_instantiate_classes(self):
        if fit_or_val(self.subcommand):
            config = self.config[self.subcommand]
            logger_args = config.logger.init_args
            logger_args.name = str(Path(logger_args.name) / f'fold-{config.fold_id}')
        super().before_instantiate_classes()

    def _run_subcommand(self, subcommand: str):
        torch.set_float32_matmul_precision(self._get(self.config, 'float32_matmul_precision'))
        if fit_or_val(subcommand):
            self.datamodule.fold_id = self._get(self.config, 'fold_id')
        super()._run_subcommand(subcommand)
