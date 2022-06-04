from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from .args import UMeIArgs

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

def build_encoder(args: UMeIArgs) -> UEncoderBase:
    if args.encoder == 'resnet':
        from monai.networks import nets
        resnet_builder = getattr(nets, f'resnet{args.model_depth}')
        model: nn.Module = resnet_builder(
            n_input_channels=args.num_input_channels,
            feed_forward=False,
            conv1_t_size=args.resnet_conv1_size,
            conv1_t_stride=args.resnet_conv1_stride,
            layer1_stride=args.resnet_layer1_stride,
            in_planes=args.base_feature_size,
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
            img_size=args.sample_shape,
            patch_size=(args.vit_patch_size, args.vit_patch_size, args.vit_patch_size),
            hidden_size=args.vit_hidden_size,
            classification=False,
        )
    elif args.encoder == 'swt':
        from monai.networks.nets.swin_unetr import SwinTransformer
        model = SwinTransformer(
            in_chans=args.num_input_channels,
            embed_dim=args.base_feature_size,
            window_size=(7, 7, 7),
            patch_size=(args.vit_patch_size, args.vit_patch_size, args.vit_patch_size),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            # mlp_ratio=4.0,
            # qkv_bias=True,
            use_checkpoint=True,
        )
        if args.pretrain_path is not None:
            # assume weights from https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/
            state_dict = {
                k[8:]: v
                for k, v in torch.load(args.pretrain_path)["state_dict"].items()
                if k.startswith('swinViT.')
            }
            miss, unexpected = model.load_state_dict(state_dict)
            assert len(miss) == 0 and len(unexpected) == 0
            print(f'load pre-trained swin-unetr encoder from {args.pretrain_path}')
        return model
    else:
        raise ValueError(f'not supported encoder: {args.encoder}')

def build_decoder(args: UMeIArgs):
    if args.decoder == 'cnn':
        from umei.models.cnn_decoder import CnnDecoder
        return CnnDecoder(base_feature_size=args.base_feature_size)
    elif args.decoder == 'sunetr':
        from monai.networks.nets import SwinUnetrDecoder
        model = SwinUnetrDecoder(args.num_input_channels, feature_size=args.base_feature_size)
        if args.pretrain_path is not None:
            # assume weights from https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/
            state_dict = {
                k: v
                for k, v in torch.load(args.pretrain_path)["state_dict"].items()
                if not k.startswith('swinViT.') and not k.startswith('out.')
            }
            miss, unexpected = model.load_state_dict(state_dict)
            assert len(miss) == 0 and len(unexpected) == 0
            print(f'load pre-trained swin-unetr decoder from {args.pretrain_path}')
        return model
    else:
        raise ValueError(f'not supported decoder: {args.decoder}')
