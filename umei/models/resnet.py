import torch

from monai.networks.nets import ResNet

from ..model import UEncoderBase, U_ENCODER_OUTPUT

class UResNetEncoder(UEncoderBase, ResNet):
    def __init__(
        self,
        *,
        num_input_channels: int,
        feature_sizes: list[int],
        layers: list[int],
    ):
        ResNet.__init__(self, n_input_channels=num_input_channels, )

    def forward(self, img: torch.FloatTensor) -> U_ENCODER_OUTPUT:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x
