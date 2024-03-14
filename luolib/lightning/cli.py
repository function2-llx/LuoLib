from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI as LightningCLIBase,
    SaveConfigCallback as SaveConfigCallbackBase,
)
from lightning.pytorch.loggers import WandbLogger
import torch

from luolib.datamodule import CrossValDataModule, ExpDataModuleBase
from luolib.utils import fall_back_none
from .module import LightningModule
from .trainer import Trainer
from .utils import OptimConf

class SaveConfigCallback(SaveConfigCallbackBase):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if self.already_saved:
            return
        if self.save_to_log_dir and trainer.log_dir is None:
            return
        return super().setup(trainer, pl_module, stage)

@dataclass
class OptimDict:
    pass

class LightningCLI(LightningCLIBase):
    _subcommand_preparing: str | None = None
    trainer: Trainer
    model: LightningModule
    datamodule: ExpDataModuleBase

    def __init__(
        self,
        *,
        model_class: type[LightningModule] | Callable[..., LightningModule] | None = LightningModule,
        datamodule_class: type[ExpDataModuleBase] | Callable[..., ExpDataModuleBase] | None = ExpDataModuleBase,
        save_config_callback: type[SaveConfigCallback] | None = SaveConfigCallback,
        save_config_kwargs: dict[str, ...] | None = None,
        trainer_class: type[Trainer] | Callable[..., Trainer] = Trainer,
        trainer_defaults: dict[str, ...] | None = None,
        seed_everything_default: bool | int = 42,
        parser_kwargs: dict[str, ...] | dict[str, dict[str, ...]] | None = None,
        subclass_mode_model: bool = True,
        subclass_mode_data: bool = True,
        auto_configure_optimizers: bool = False,
        optim_dict_class: type[OptimDict] | None = None,
        **kwargs,
    ):
        """
        Args:
            optim_dict_class: key -> OptimizationConf
        """
        save_config_kwargs = fall_back_none(save_config_kwargs, {'config_filename': 'conf.yaml'})
        if trainer_defaults is None:
            trainer_defaults = {
                'callbacks': [
                    LearningRateMonitor(),
                    ModelSummary(max_depth=2),
                ]
            }
        parser_kwargs = fall_back_none(parser_kwargs, {'parser_mode': "omegaconf"})
        self.optim_dict_class = optim_dict_class
        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
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

    @property
    def active_config(self):
        return self.config if self.subcommand is None else self.config[self.subcommand]

    @property
    def active_config_init(self):
        return self.config_init if self.subcommand is None else self.config_init[self.subcommand]

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

    @property
    def is_preparing_fit(self):
        return self._subcommand_preparing in {'fit', 'play'}

    @property
    def model_prefix(self):
        return 'model.init_args' if self.subclass_mode_model else 'model'

    @property
    def data_prefix(self):
        return 'data.init_args' if self.subclass_mode_data else 'data'

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--float32_matmul_precision', type=Literal['medium', 'high', 'highest'], default='medium')
        parser.link_arguments('trainer.max_steps', f'{self.data_prefix}.dataloader.num_batches')
        parser.add_argument('--compile', type=bool, default=True)
        parser.add_argument('--trace_numpy', type=bool, default=False)
        if self.is_preparing_fit:
            parser.add_argument('--logger', type=WandbLogger, enable_path=True)
            parser.link_arguments('logger', 'trainer.logger', apply_on='instantiate')
        else:
            parser.add_argument('--logger', type=Literal[False], default=False)
            parser.link_arguments('logger', 'trainer.logger')
        parser.add_argument('--mp_start_method', type=Literal['fork', 'spawn', 'forkserver'], default='fork')
        parser.add_argument('--mp_sharing_strategy', type=Literal['file_descriptor', 'file_system'], default='file_descriptor')
        if self.is_preparing_fit:
            if self.optim_dict_class is None:
                parser.add_dataclass_arguments(OptimConf, 'optim')
            else:
                parser.add_dataclass_arguments(self.optim_dict_class, 'optim')
        super().add_arguments_to_parser(parser)

    def before_instantiate_classes(self):
        config = self.active_config
        if self.subcommand in {'fit', 'validate', 'play'}:
            logger_args = config.logger.init_args
            save_dir = Path(logger_args.save_dir) / logger_args.name / f'seed-{config.seed_everything}'
            # wandb wants to use a directory already existing: https://github.com/wandb/wandb/issues/714#issuecomment-565870686
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            logger_args.save_dir = str(save_dir)
        # tqdm (and so on) may initialize multiprocessing context
        torch.multiprocessing.set_start_method(config.mp_start_method)
        torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)
        torch.set_float32_matmul_precision(config.float32_matmul_precision)
        super().before_instantiate_classes()

    def fit(self, model: LightningModule, **kwargs):
        optim: OptimConf | OptimDict = self.active_config_init.optim
        if isinstance(optim, OptimConf):
            model.optim = [optim]
        else:
            model.optim = [optim_conf for optim_conf in vars(optim).values() if isinstance(optim_conf, OptimConf)]
        # https://github.com/Lightning-AI/lightning/issues/17283
        if self._get(self.config, 'compile'):
            # https://github.com/pytorch/pytorch/issues/112335
            from torch._dynamo import config
            config.trace_numpy = self._get(self.config, 'trace_numpy')
            model = torch.compile(model)
        self.trainer.fit(model, **kwargs)

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
        if self._subcommand_preparing in {'fit', 'validate', 'play'}:
            parser.add_argument('fold_id', type=int)
        super().add_arguments_to_parser(parser)

    def before_instantiate_classes(self):
        if self.subcommand in {'fit', 'validate', 'play'}:
            config = self.config[self.subcommand]
            logger_args = config.logger.init_args
            logger_args.name = str(Path(logger_args.name) / f'fold-{config.fold_id}')
        super().before_instantiate_classes()

    def _run_subcommand(self, subcommand: str):
        torch.set_float32_matmul_precision(self._get(self.config, 'float32_matmul_precision'))
        if subcommand in {'fit', 'validate', 'play'}:
            self.datamodule.fold_id = self._get(self.config, 'fold_id')
        super()._run_subcommand(subcommand)
