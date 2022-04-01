from dataclasses import dataclass, field

import torch
from torch import nn

@dataclass
class UEncoderOutput:
    cls_feature: torch.FloatTensor = None
    feature_maps: list[torch.FloatTensor] = field(default_factory=list)

class UEncoderBase(nn.Module):
    def forward(self, img: torch.FloatTensor) -> UEncoderOutput:
        raise NotImplementedError

    @property
    def cls_feature_size(self) -> int:
        raise NotImplementedError

@dataclass
class UDecoderOutput:
    feature_maps: list[torch.FloatTensor]

class UDecoderBase(nn.Module):
    def forward(self, encoder_feature_maps: list[torch.FloatTensor]) -> list[torch.FloatTensor]:
        raise not NotImplementedError

    @property
    def feature_sizes(self) -> list[int]:
        raise NotImplementedError