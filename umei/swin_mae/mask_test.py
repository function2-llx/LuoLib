from einops import rearrange
import torch
from tqdm import trange

from umei.swin_mae.model import MaskSwin
from umei.swin_mae.args import SwinMAEArgs

def main():
    args: SwinMAEArgs = SwinMAEArgs.from_yaml_file('../../conf/amos/t1/swin_mae-96x96.yml')
    mask_swin = MaskSwin(
        mask_ratio=args.mask_ratio,
        block_shape=args.mask_block_shape,
        in_chans=args.num_input_channels,
        base_feature_size=24,
        window_size=args.swin_window_size,
        patch_shape=args.vit_patch_shape,
        depths=args.vit_depths,
        num_heads=args.vit_num_heads,
        # mlp_ratio=4.0,
        # qkv_bias=True,
        use_checkpoint=True,
    ).cuda().eval()

    print('expected mask ratio:', args.mask_ratio)
    with torch.no_grad():
        for _ in trange(1000):
            x = torch.randn(4, 1, 96, 96, 96, device='cuda')
            print(mask_swin(x))

if __name__ == '__main__':
    main()
