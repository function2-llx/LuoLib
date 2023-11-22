from omegaconf import OmegaConf

__all__ = []

OmegaConf.register_new_resolver('divide', lambda a, b: a / b)
OmegaConf.register_new_resolver('eval', lambda x: eval(x))
