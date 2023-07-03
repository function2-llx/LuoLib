import importlib

from omegaconf import OmegaConf

from luolib.utils import PathLike

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_conf(conf: dict | PathLike):
    if isinstance(conf, PathLike):
        conf = OmegaConf.load(conf)
    return get_obj_from_str(conf['target'])(**conf.get('kwargs', dict()))
