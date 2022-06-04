from dataclasses import dataclass, field

import torch
from torch import nn

# since monai models are adapted to umei API (relying on umei),
# don't import monai globally or will lead to circular import

@dataclass
class UEncoderOutput:
    cls_feature: torch.FloatTensor = None
    hidden_states: list[torch.FloatTensor] = field(default_factory=list)

class UEncoderBase(nn.Module):
    def forward(self, img: torch.FloatTensor) -> UEncoderOutput:
        raise NotImplementedError

@dataclass
class UDecoderOutput:
    feature_maps: list[torch.FloatTensor]

class UDecoderBase(nn.Module):
    def forward(self, img: torch.FloatTensor, encoder_hidden_states: list[torch.FloatTensor]) -> UDecoderOutput:
        raise not NotImplementedError
