from pathlib import Path
from types import SimpleNamespace

from pytorch_lightning import LightningModule
from timm.optim.optim_factory import param_groups_layer_decay
from timm.scheduler.scheduler import Scheduler
import torch
from torch import nn
from torch.optim import Optimizer

from monai.umei import Backbone
from monai.utils import ensure_tuple

from umei.conf import ExpConfBase
from ..registry import backbone_registry
from ..utils import create_model

__all__ = [
    'ExpModelBase',
]

from umei.types import ParameterGroup

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

    def no_weight_decay(self):
        # modify from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py, `configure_optimizers`
        from torch.nn.modules.conv import _ConvNd
        whitelist_weight_modules = (
            nn.Linear,
            _ConvNd,
        )
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        blacklist_weight_modules = (
            nn.LayerNorm,
            _BatchNorm,
            _InstanceNorm,
            nn.Embedding,
        )
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            if hasattr(m, 'no_weight_decay'):
                for pn in m.no_weight_decay():
                    no_decay.add(f'{mn}.{pn}' if mn else pn)

            for pn, p in m.named_parameters(prefix=mn, recurse=False):
                if not p.requires_grad:
                    continue
                if pn.endswith('.bias'):
                    # all biases will not be decayed
                    no_decay.add(pn)
                elif pn.endswith('.weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(pn)
                else:
                    assert pn.endswith('.weight') and isinstance(m, blacklist_weight_modules)
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(pn)

        inter_params = decay & no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        return no_decay

    def get_parameter_groups(self) -> list[ParameterGroup]:
        optim = self.conf.optimizer
        return param_groups_layer_decay(self, optim.weight_decay, self.no_weight_decay(), optim.layer_decay)
        # if self.trainer.is_global_zero:
        #     obj = {'decay': decay, 'no decay': no_decay}
        #     (self.log_exp_dir / 'parameters.json').write_text(json.dumps(obj, indent=4, ensure_ascii=False))
        #

    def configure_optimizers(self):
        from timm.optim import create_optimizer_v2
        optim = self.conf.optimizer
        parameter_groups = self.get_parameter_groups()
        # timm's typing is really not so great
        optimizer = create_optimizer_v2(
            parameter_groups,
            optim.name,
            optim.lr,
            optim.weight_decay,
            optim.layer_decay,
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
