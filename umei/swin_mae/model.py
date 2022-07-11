import pytorch_lightning as pl
import torch
from torch import nn

from monai.networks.nets import SwinTransformer, SwinUnetrDecoder
from umei.swin_mae.args import SwinMAEArgs

class SwinMAE(pl.LightningModule):
    def __init__(self, args: SwinMAEArgs):
        super().__init__()
        self.args = args

        self.encoder = SwinTransformer(
            in_chans=args.num_input_channels,
            embed_dim=args.base_feature_size,
            window_size=args.swin_window_size,
            patch_size=args.vit_patch_shape,
            depths=args.vit_depths,
            num_heads=args.vit_num_heads,
            # mlp_ratio=4.0,
            # qkv_bias=True,
            use_checkpoint=True,
        )

        self.decoder = SwinUnetrDecoder(args.num_input_channels, feature_size=args.base_feature_size)

        with torch.no_grad():
            dummy_input = torch.zeros(1, args.num_input_channels, *args.sample_shape)
            dummy_encoder_output = self.encoder.forward(dummy_input)
            dummy_decoder_output = self.decoder.forward(dummy_input, dummy_encoder_output.hidden_states)
            decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.base_feature_size))

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Convnd)
        w: torch.Tensor = self.encoder.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w)


    def training_step(self, x: torch.Tensor, *args, **kwargs):
        pass
