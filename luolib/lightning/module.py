from __future__ import annotations as _

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
import json
from typing import final

from lightning import LightningDataModule, LightningModule as _LightningModuleBase
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning_utilities import apply_to_collection
from lightning_utilities.core.rank_zero import rank_prefixed_message
from peft import PeftModel
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
import torch
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer

from luolib import lightning as lpl
from luolib.optim import infer_weight_decay_keys
from luolib.scheduler import HybridScheduler
from luolib.utils.grad import compute_grad_norm
from .utils import OptimConf, build_single_optim

__all__ = [
    'LightningModule',
]

from ..utils import fall_back_none

@dataclass
class TrainingStepContext:
    batch: ... = None

class LightningModule(_LightningModuleBase):
    trainer: lpl.Trainer
    check_grad: bool = False

    def __init__(
        self, *,
        log_grad_norm: bool = True,
        check_grad: bool | None = None,
        **kwargs,
    ):
        # TODO: should I move log_grad_norm to some callback?
        super().__init__(**kwargs)
        self.log_grad_norm = log_grad_norm
        if check_grad is not None:
            self.check_grad = check_grad
        self.training_step_context = TrainingStepContext()

    def get_decay_keys(self) -> set[str]:
        return infer_weight_decay_keys(self)

    @property
    def peft_model(self) -> PeftModel:
        return self._peft_model[0]

    @property
    def peft_model_prefix(self):
        return self._peft_model_prefix

    # @peft_model.setter
    # def peft_model(self, value):
    #     self._peft_model = value

    def set_peft_model(self, value: PeftModel, prefix: str = ''):
        """
        Args:
            prefix: must end with "." if nonempty
        Returns:
        """
        self._peft_model = (value, )
        prefix_remove_dot = prefix[:-1] if prefix.endswith('.') else prefix
        assert self.get_submodule(prefix_remove_dot) is value.base_model.model
        prefix_with_dot = prefix if prefix.endswith('.') else f'{prefix}.'
        self._peft_model_prefix = prefix_with_dot

    @property
    def batch(self):
        """The current batch"""
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

    def __setattr__(self, name: str, value: ...) -> None:
        if name == 'peft_model':
            # let nn.Module not register it as a submodule
            value = (value, )
        super().__setattr__(name, value)

    @cache
    @final
    def _get_decay_keys(self) -> set[str]:
        return self.get_decay_keys()

    @property
    def optims(self):
        return self._optim

    @optims.setter
    def optims(self, optim: dict[str, OptimConf]):
        self._optim = optim

    def configure_optimizers(self):
        optimizer, lr_scheduler_config, param_groups = build_single_optim(
            self, self.optims, self._get_decay_keys(), self.trainer,
        )
        (self.trainer.log_dir / 'optim.json').write_text(
            json.dumps(
                {
                    param_group['name']: [pn for pn, _ in param_group['params']]
                    for param_group in param_groups
                },
                indent=4,
                ensure_ascii=False,
            ),
        )
        return {
            'optimizer': optimizer,
            # call `vars` due to: https://github.com/Lightning-AI/lightning/issues/18870
            'lr_scheduler': vars(lr_scheduler_config),
        }

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.trainer.log_dir / 'model.txt').write_text(repr(self))

    def on_train_batch_start(self, batch: ..., batch_idx: int) -> int | None:
        super().on_train_batch_start(batch, batch_idx)
        self.training_step_context.batch = batch
        return None  # make PyCharm happy

    def lr_scheduler_step(self, scheduler: ..., metric=None):
        match scheduler:
            case HybridScheduler():
                for inner_scheduler in scheduler._schedulers:
                    self.lr_scheduler_step(inner_scheduler, metric)
            case TIMMScheduler():
                scheduler.step_update(self.global_step + 1, metric)
            case _:
                super().lr_scheduler_step(scheduler, metric)

    def on_after_backward(self):
        if self.check_grad and self.trainer.world_size > 1:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(rank_prefixed_message(f'none grad: {name}', self.global_rank))

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: GradClipAlgorithmType | None = None,
    ) -> None:
        # NOTE: PL's design for gradient clipping is incomplete, and I don't have time to be comprehensive enough, either
        #   see https://github.com/Lightning-AI/pytorch-lightning/issues/19235
        gradient_clip_algorithm = fall_back_none(gradient_clip_algorithm, self.trainer.gradient_clip_algorithm)
        if gradient_clip_algorithm == GradClipAlgorithmType.NORM and isinstance(self.trainer.model, FullyShardedDataParallel):
            grad_norm = self.trainer.model.clip_grad_norm_(gradient_clip_val)
            grad_norm_clipped = min(grad_norm, gradient_clip_val)
        else:
            grad_norm = compute_grad_norm(self)
            self.clip_gradients(
                optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm,
            )
            grad_norm_clipped = compute_grad_norm(self)
        if self.log_grad_norm:
            # log gradient before gradient clipping
            self.log('grad_norm', grad_norm)
            self.log('grad_norm-clipped', grad_norm_clipped)
        # # save bad state for diagnosis
        # # TODO: also check:
        # #  - loss, but I can't get the step output, over-engineering, 作茧自缚了 :( PL should really officially support "training step context"
        # #  - parameters after optimizer step, but this seems to require save the state before optimization
        # if (save_dir := self.trainer.log_dir / 'bad-state' / f'rank-{self.global_rank}').exists():
        #     return
        # for param in self.parameters():
        #     if param.grad is not None and not param.isfinite().all():
        #         save_dir.mkdir(parents=True)
        #         self.trainer.save_checkpoint(save_dir, local=True)
        #         torch.save(self.training_step_context.batch, save_dir / f'batch.pt')
        #         break

    @property
    def datamodule(self) -> LightningDataModule:
        return self.trainer.datamodule

    def all_gather(self, *args, **kwargs):
        ret = super().all_gather(*args, **kwargs)
        if self.trainer.world_size == 1:
            # let me do it for you: https://github.com/Lightning-AI/pytorch-lightning/issues/19195
            ret = apply_to_collection(ret, torch.Tensor, lambda x: x[None])
        return ret

    def log_dict(
        self,
        data: Mapping[str, ...],
        *args,
        sync_dist: bool = False,
        **kwargs,
    ) -> None:
        """reduce sync_dist cost"""
        if not sync_dist or not all(isinstance(value, (int, float, torch.Tensor)) for value in data.values()):
            return super().log_dict(data, *args, sync_dist=sync_dist, **kwargs)
        data = dict(data)
        values = torch.tensor([*data.values()])
        # NOTE: make sure that len(values) are consistent across processes
        values = self.all_gather(values)
        values = values.mean(dim=0)
        for i, name in enumerate(data):
            data[name] = values[i]
        super().log_dict(data, *args, sync_dist=False, **kwargs)
