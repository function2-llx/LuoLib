import torch
from torch import nn
import pytorch_lightning as pl

from umei.models.swin import SwinBackbone, SwinLayer

def main():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # swin = SwinBackbone(
    #     in_channels=1,
    #     layer_channels=[24, 48, 96, 192, 384, 768],
    #     kernel_sizes=[
    #         [3, 3, 3],
    #         [3, 3, 3],
    #         [6, 6, 6],
    #         [6, 6, 6],
    #         [6, 6, 6],
    #         [6, 6, 6],
    #     ],
    #     layer_depths=[1, 2, 2, 2, 6, 2],
    #     num_conv_layers=2,
    #     num_heads=[1, 1, 3, 6, 12, 24],
    #     drop_path_rate=.1
    # )

    for stage in range(4):
        # stage = 1
        dim = 48 << stage
        batch_size = 1
        shape = (batch_size, dim, 48 >> stage, 48 >> stage, 48 >> stage)
        depth = 6
        num_heads = 3 << stage
        window_size = [6, 6, 6]
        x = torch.randn(*shape)

        pl.seed_everything(42)
        layer1 = SwinLayer(
            dim,
            depth,
            num_heads,
            window_size,
        )
        o1 = layer1(x)
        from monai.networks.nets.swin_unetr import BasicLayer
        pl.seed_everything(42)
        layer2 = BasicLayer(
            dim,
            depth,
            num_heads,
            window_size,
            drop_path=[0.] * depth,
        )
        o2 = layer2(x)
        print(o1.numel())
        print((o1 != o2).sum())
    # sd1 = layer1.state_dict()
    # sd2 = layer2.state_dict()
    # for k in sd1.keys():
    #     v1 = sd1[k]
    #     v2 = sd2[k]
    #     if (v1.view(-1) != v2.view(-1)).sum() != 0:
    #         print(k)

if __name__ == '__main__':
    main()
