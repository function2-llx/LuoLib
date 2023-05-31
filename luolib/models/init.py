import numpy as np
from torch import nn

def init_common(m: nn.Module):
    match type(m):
        case nn.Linear:
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        case nn.modules.conv._ConvNd:
            # Kaiming normal (ReLU) with fix group fanout
            fan_out = np.prod(m.kernel_size) * m.out_channels // m.groups
            nn.init.normal_(m.weight, 0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        case nn.Embedding:
            nn.init.trunc_normal_(m.weight)
