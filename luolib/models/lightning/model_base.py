from pathlib import Path

from pytorch_lightning import LightningModule
from timm.optim.optim_factory import param_groups_layer_decay
from timm.scheduler.scheduler import Scheduler
import torch
from torch.optim import Optimizer

from monai.luolib import Backbone
from monai.utils import ensure_tuple

from luolib.conf import ExpConfBase
from luolib.types import ParamGroup
from luolib.utils import partition_by_predicate
from luolib.optim import create_optimizer
from luolib.scheduler import create_scheduler
from ..registry import backbone_registry
from ..utils import create_model, get_no_weight_decay_keys

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

    def get_param_groups(self) -> list[ParamGroup]:
        others_no_decay_keys, backbone_no_decay_keys = map(
            set,
            partition_by_predicate(lambda k: k.startswith('backbone.'), get_no_weight_decay_keys(self)),
        )
        backbone_optim = self.conf.backbone_optim
        optim = self.conf.optimizer
        param_groups = []
        if backbone_optim.lr != 0:
            backbone_param_groups: list[ParamGroup] = param_groups_layer_decay(
                self.backbone,
                backbone_optim.weight_decay,
                backbone_no_decay_keys,
                backbone_optim.layer_decay,
            )
            for param_group in backbone_param_groups:
                param_group['lr'] = backbone_optim.lr * param_group.pop('lr_scale')
            param_groups.extend(backbone_param_groups)

        others_decay_params, others_no_decay_params = map(
            lambda nps: map(lambda np: np[1], nps),  # remove names
            partition_by_predicate(
                lambda np: np[0] in others_no_decay_keys,
                filter(lambda np: not np[0].startswith('backbone.'), self.named_parameters()),
            )
        )

        param_groups.extend([
            {
                'params': others_decay_params,
                'weight_decay': optim.weight_decay,
            },
            {
                'params': others_no_decay_params,
                'weight_decay': 0.,
            }
        ])

        return param_groups

    def configure_optimizers(self):
        conf = self.conf
        optimizer = create_optimizer(conf.optimizer, self.get_param_groups())
        scheduler = create_scheduler(conf.scheduler, optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': conf.scheduler.interval,
                'frequency': conf.scheduler.frequency,
                'reduce_on_plateau': conf.scheduler.reduce_on_plateau,
                'monitor': conf.monitor,
            },
        }

    def lr_scheduler_step(self, scheduler: Scheduler, metric):
        # make compatible with timm scheduler
        conf = self.conf
        match conf.scheduler.interval:
            case 'epoch':
                scheduler.step(self.current_epoch + 1, metric)
            case 'step':
                from timm.scheduler import PlateauLRScheduler
                if isinstance(scheduler, PlateauLRScheduler):
                    scheduler.step(self.global_step // conf.scheduler.frequency, metric)
                else:
                    scheduler.step_update(self.global_step, metric)

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

    @property
    def interpolate_mode(self):
        match self.conf.spatial_dims:
            case 2:
                return 'bilinear'
            case 3:
                return 'trilinear'
            case _:
                raise ValueError
