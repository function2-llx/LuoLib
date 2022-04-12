from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from umei.utils import UMeIArgs

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

def build_encoder(args: UMeIArgs) -> UEncoderBase:
    from monai.networks import nets
    if args.model_name == 'resnet':
        resnet_builder = getattr(nets, f'resnet{args.model_depth}')
        model: nn.Module = resnet_builder(
            n_input_channels=args.num_input_channels,
            feed_forward=False,
            conv1_t_size=args.resnet_conv1_size,
            conv1_t_stride=args.resnet_conv1_stride,
            shortcut_type=args.resnet_shortcut,
        )

        if args.pretrain_path is not None:
            # assume pre-trained weights are from https://github.com/Tencent/MedicalNet
            dp_model = nn.DataParallel(model)
            state_dict = dp_model.state_dict()
            pretrain_state_dict = torch.load(args.pretrain_path, map_location='cpu')['state_dict']
            state_dict.update({
                k: v for k, v in pretrain_state_dict.items()
                if k in state_dict and k != 'module.conv1.weight'
            })
            dp_model.load_state_dict(state_dict)
            model: nets.ResNet = dp_model.module  # type: ignore

            # handle number of input channels that is possible different from the pre-trained model
            for attr in ['weight', 'bias']:
                param: Optional[nn.Parameter] = getattr(model.conv1, attr, None)
                pretrain_param_data: Optional[torch.Tensor] = getattr(pretrain_state_dict, f'module.conv1.{attr}', None)
                if param is not None and pretrain_param_data is not None:
                    param.data = pretrain_param_data.repeat(1, args.num_input_channels)
        return model
    else:
        raise NotImplementedError
