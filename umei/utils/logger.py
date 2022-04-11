from typing import Optional

from pytorch_lightning.loggers import WandbLogger

# cannot wait: https://github.com/PyTorchLightning/pytorch-lightning/pull/12604/
class MyWandbLogger(WandbLogger):
    @WandbLogger.name.getter
    def name(self) -> Optional[str]:
        return self._experiment.name if self._experiment else self._name
