import numpy as np
from torch import nn
from torch.nn.init import trunc_normal_

def init_linear_conv(m: nn.Module):
    match type(m):
        case nn.Linear:
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        case nn.Conv3d | nn.ConvTranspose3d:
            # Kaiming normal (ReLU) with fix group fanout
            fan_out = np.prod(m.kernel_size) * m.out_channels // m.groups
            nn.init.normal_(m.weight, 0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
