from typing import Optional

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn.functional import interpolate

from .args import UMeIArgs
from .model import UDecoderOutput, UEncoderBase, UEncoderOutput

class UMeI(LightningModule):
    cls_loss_fn: nn.Module
    seg_loss_fn: nn.Module

    def __init__(self, args: UMeIArgs, *, has_decoder: bool):
        super().__init__()
        self.args = args
        self.encoder = build_encoder(args)
        with torch.no_grad():
            self.encoder.eval()
            dummy_input = torch.zeros(1, args.num_input_channels, *args.sample_shape)
            dummy_encoder_output = self.encoder.forward(dummy_input)
            encoder_feature_sizes = [feature.shape[1] for feature in dummy_encoder_output.hidden_states]
        if self.args.num_cls_classes is not None:
            encoder_cls_feature_size = dummy_encoder_output.cls_feature.shape[1]
            self.cls_head = nn.Linear(encoder_cls_feature_size + args.clinical_feature_size, args.num_cls_classes)
            nn.init.constant_(torch.as_tensor(self.cls_head.bias), 0)

        self.decoder = None
        if has_decoder:
            self.decoder = build_decoder(args, encoder_feature_sizes)
            with torch.no_grad():
                dummy_decoder_output = self.decoder.forward(dummy_input, dummy_encoder_output.hidden_states)
                decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]
            from monai.networks.blocks import UnetOutBlock
            # i-th seg head for the last i-th output from decoder
            self.seg_heads = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=3,
                    in_channels=decoder_feature_sizes[-i - 1],
                    out_channels=args.num_seg_classes,
                )
                for i in range(args.num_seg_heads)
            ])

    def training_step(self, batch: dict, *args, **kwargs) -> STEP_OUTPUT:
        img = batch[self.args.img_key]
        encoder_out: UEncoderOutput = self.encoder(img)
        ret = {'loss': torch.tensor(0., device=self.device)}
        if self.args.cls_key in batch:
            cls_out = self.cls_head(torch.cat((encoder_out.cls_feature, batch[self.args.clinical_key]), dim=1))
            cls_loss = self.cls_loss_fn(cls_out, batch[self.args.cls_key])
            # self.log('cls_loss', cls_loss, prog_bar=True)
            ret['loss'] += cls_loss * self.args.cls_loss_factor
            ret['cls_loss'] = cls_loss
            ret['cls_logit'] = cls_out
        if self.decoder is not None and self.args.seg_key in batch:
            seg_label: torch.IntTensor = batch[self.args.seg_key]
            decoder_out: UDecoderOutput = self.decoder.forward(img, encoder_out.hidden_states)
            seg_loss = torch.sum(torch.stack([
                self.seg_loss_fn(
                    interpolate(seg_head(feature_map), seg_label.shape[2:], mode='trilinear'),
                    seg_label
                ) / 2 ** i
                for i, (feature_map, seg_head) in enumerate(zip(decoder_out.feature_maps[::-1], self.seg_heads))
            ]))
            ret['loss'] += seg_loss * self.args.seg_loss_factor
            ret['seg_loss'] = seg_loss
        for k in ['cls_loss', 'seg_loss']:
            if k in ret:
                self.log(f'train/{k}', ret[k])
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder.forward(x)
        if self.decoder is None:
            return output.cls_feature
        fm = self.decoder.forward(x, output.hidden_states).feature_maps[-1]
        return self.seg_heads[0](fm)

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
            print(f'load pre-trained med-3d weights from {args.pretrain_path}')
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

def build_decoder(args: UMeIArgs, encoder_feature_sizes: list[int]):
    if args.decoder == 'cnn':
        from umei.models.cnn_decoder import CnnDecoder
        return CnnDecoder(encoder_feature_sizes, out_channels=args.base_feature_size // 2)
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
