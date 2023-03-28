import operator
from pathlib import Path
from types import SimpleNamespace

import cytoolz
from pytorch_lightning import LightningModule
from timm.optim.optim_factory import param_groups_layer_decay
from timm.scheduler.scheduler import Scheduler
import torch
from torch.optim import Optimizer

from monai.umei import Backbone
from monai.utils import ensure_tuple

from umei.conf import ExpConfBase
from ..registry import backbone_registry
from ..utils import create_model, get_no_weight_decay_keys

__all__ = [
    'ExpModelBase',
]

from umei.types import ParamGroup

class SimpleReprMixin(object):
    """A mixin implementing a simple __repr__."""
    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

class ExpModelBase(LightningModule):
    def __init__(self, conf: ExpConfBase):
        super().__init__()
        self.conf = conf
        self.backbone = self.create_backbone()

    def create_backbone(self) -> Backbone:
        return create_model(self.conf.backbone, backbone_registry)

    def backbone_dummy(self):
        with torch.no_grad():
            self.backbone.eval()
            dummy_input = torch.zeros(1, self.conf.num_input_channels, *self.conf.sample_shape)
            dummy_output = self.backbone.forward(dummy_input)
            print('backbone output shapes:')
            for x in dummy_output.feature_maps:
                print(x.shape)
        return dummy_input, dummy_output

    @property
    def tta_flips(self):
        match self.conf.spatial_dims:
            case 2:
                return [[2], [3], [2, 3]]
            case 3:
                return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
            case _:
                raise ValueError

    @property
    def log_exp_dir(self) -> Path:
        assert self.trainer.is_global_zero
        from pytorch_lightning.loggers import WandbLogger
        logger: WandbLogger = self.trainer.logger   # type: ignore
        return Path(logger.experiment.dir)

    def on_fit_start(self):
        if not self.trainer.is_global_zero:
            return
        with open(self.log_exp_dir / 'fit-summary.txt', 'w') as f:
            print(self, file=f, end='\n\n\n')
            print('optimizers:\n', file=f)
            for optimizer in ensure_tuple(self.optimizers()):
                print(optimizer, file=f)
            print('\n\n', file=f)
            print('schedulers:\n', file=f)
            for scheduler in ensure_tuple(self.lr_schedulers()):
                print(scheduler, file=f)

    def get_parameter_groups(self) -> list[ParamGroup]:
        backbone_no_decay_keys, others_no_decay_keys = map(
            set,
            operator.itemgetter(True, False)(
                cytoolz.groupby(lambda k: k.startswith('backbone.'), get_no_weight_decay_keys(self))
            ),
        )
        optim = self.conf.optimizer
        backbone_param_groups: list[ParamGroup] = param_groups_layer_decay(
            self.backbone,
            optim.weight_decay,
            backbone_no_decay_keys,
            optim.layer_decay,
        )
        for param_group in backbone_param_groups:
            param_group['lr'] = self.conf.backbone_lr * param_group.pop('lr_scale')

        others_decay_params, others_no_decay_params = map(
            lambda nps: map(lambda np: np[1], nps),
            operator.itemgetter(False, True)(
                cytoolz.groupby(
                    lambda n, _: n in others_no_decay_keys,
                    filter(lambda n, _: not n.startswith('backbone.'), self.named_parameters()),
                ),
            ),
        )

        return backbone_param_groups + [
            {
                'params': others_decay_params,
                'weight_decay': optim.weight_decay,
            },
            {
                'params': others_no_decay_params,
                'weight_decay': 0.,
            }
        ]

    def configure_optimizers(self):
        from timm.optim import create_optimizer_v2
        optim = self.conf.optimizer
        parameter_groups = self.get_parameter_groups()
        # timm's typing is really not so great
        optimizer = create_optimizer_v2(
            parameter_groups,
            optim.name,
            optim.lr,
            **optim.kwargs,
        )

        from timm.scheduler import create_scheduler
        scheduler, _num_epochs = create_scheduler(SimpleNamespace(**self.conf.scheduler), optimizer)
        if type(scheduler).__repr__ == object.__repr__:
            type(scheduler).__repr__ = SimpleReprMixin.__repr__
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.conf.monitor,
        }

    def lr_scheduler_step(self, scheduler: Scheduler, metric):
        # make compatible with timm scheduler
        return scheduler.step(self.current_epoch + 1, metric)

    def optimizer_zero_grad(self, _epoch, _batch_idx, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=self.conf.optimizer_set_to_none)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        grad_norm = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(g.detach())
                for p in self.parameters() if (g := p.grad) is not None
            ])
        )
        self.log('train/grad_norm', grad_norm)
