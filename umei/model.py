from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn

U_ENCODER_OUTPUT = list[torch.Tensor]

class UEncoderBase(nn.Module):
    def forward(self, img: torch.FloatTensor) -> U_ENCODER_OUTPUT:
        pass

    @property
    def feature_sizes(self) -> list[int]:
        raise NotImplementedError

class UDecoderBase(nn.Module):
    def __init__(self):
        super().__init__()


class UMeI(LightningModule):
    def __init__(self, encoder: UEncoderBase, decoder: UDecoderBase, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def configure_optimizers(self):
        pass

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        pass
