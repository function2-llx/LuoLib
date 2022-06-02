from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from umei.utils import UMeIArgs

# since monai models are adapted to umei API (relying on umei),
# don't import monai globally or will lead to circular import

@dataclass
class UEncoderOutput:
    cls_feature: torch.FloatTensor = None
    hidden_states: list[torch.FloatTensor] = field(default_factory=list)

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
    def forward(self, img: torch.FloatTensor, encoder_hidden_states: list[torch.FloatTensor]) -> list[torch.FloatTensor]:
        raise not NotImplementedError

def build_encoder(args: UMeIArgs) -> UEncoderBase:
    if args.encoder == 'resnet':
        from monai.networks import nets
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
    elif args.encoder == 'vit':
        from monai.networks.nets import ViT
        return ViT(
            in_channels=args.num_input_channels,
            img_size=(args.sample_size, args.sample_size, args.sample_slices),
            patch_size=(args.vit_patch_size, args.vit_patch_size, args.vit_patch_size),
            hidden_size=args.vit_hidden_size,
            classification=False,
        )
    elif args.encoder == 'swt':
        from monai.networks.nets.swin_unetr import SwinTransformer
        return SwinTransformer(
            in_chans=args.num_input_channels,
            embed_dim=args.vit_hidden_size,
            window_size=(7, 7, 7),
            patch_size=(args.vit_patch_size, args.vit_patch_size, args.vit_patch_size),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            # mlp_ratio=4.0,
            # qkv_bias=True,
        )
    else:
        raise ValueError(f'not supported encoder: {args.encoder}')

def build_decoder(args: UMeIArgs):
    from monai.networks.nets.swin_unetr import UnetrUp

    if args.decoder == 'unetr':
        return UnetrUp(args.num_input_channels, feature_size=args.base_feature_size)
    else:
        raise ValueError(f'not supported encoder: {args.encoder}')
