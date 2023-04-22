from collections.abc import Iterable

from timm.optim import create_optimizer_v2
import torch

from luolib.conf import OptimizerConf
from luolib.types import ParamGroup

def create_optimizer(conf: OptimizerConf, param_groups: list[ParamGroup] | Iterable[torch.Tensor]):
    # timm's typing is really not so great
    return create_optimizer_v2(
        param_groups,
        conf.name,
        conf.lr,
        conf.weight_decay,
        **conf.kwargs,
    )
