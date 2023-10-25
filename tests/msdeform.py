import torch

from luolib.models import MultiscaleDeformablePixelDecoder

def main():
    bs = 2
    sp = 3
    spatial_shapes = [
        [48] * sp,
        [24] * sp,
        [12] * sp,
        [6] * sp,
    ]
    feature_channels = [128, 256, 512, 1024]
    feature_maps = [
        torch.randn(bs, c, *spatial_shape, device='cuda')
        for c, spatial_shape in zip(feature_channels, spatial_shapes)
    ]
    decoder = MultiscaleDeformablePixelDecoder(
        sp, feature_channels, 256, 16, 8,
    ).cuda()
    print(decoder)
    decoder(feature_maps)

if __name__ == '__main__':
    main()
