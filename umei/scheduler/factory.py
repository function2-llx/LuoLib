from timm.scheduler import CosineLRScheduler
from torch.optim import Optimizer

from umei.conf import SchedulerConf
from umei.utils import SimpleReprMixin

registry = {
    'cosine': CosineLRScheduler,
}

def create_scheduler(conf: SchedulerConf, optimizer: Optimizer):
    builder = registry[conf.name]
    scheduler = builder(optimizer, **conf.kwargs)
    if type(scheduler).__repr__ == object.__repr__:
        type(scheduler).__repr__ = SimpleReprMixin.__repr__
    return scheduler
