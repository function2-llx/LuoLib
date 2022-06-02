import torch

from umei import UDecoderBase, UEncoderBase, UMeI
from umei.datasets.amos import AmosArgs

from monai.losses import DiceFocalLoss

class AmosModel(UMeI):
    def __init__(self, args: AmosArgs, encoder: UEncoderBase, decoder: UDecoderBase):
        super().__init__(args, encoder, decoder)
        self.seg_loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True)

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        pass
