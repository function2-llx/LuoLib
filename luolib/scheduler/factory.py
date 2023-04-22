from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
from torch.optim import Optimizer

from luolib.conf import SchedulerConf
from luolib.utils import SimpleReprMixin

registry = {
    'cosine': CosineLRScheduler,
    'multistep': MultiStepLRScheduler,
}

def create_scheduler(conf: SchedulerConf, optimizer: Optimizer):
    builder = registry[conf.name]
    scheduler = builder(optimizer, **conf.kwargs)
    if type(scheduler).__repr__ == object.__repr__:
        type(scheduler).__repr__ = SimpleReprMixin.__repr__
    return scheduler
