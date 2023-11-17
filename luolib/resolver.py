from omegaconf import OmegaConf

__all__ = []

OmegaConf.register_new_resolver('divide', lambda a, b: a / b)
