from torch import nn

class NoWeightDecayParameter(nn.Parameter):
    """explicitly indicate that a parameter requires no weight decay"""
    pass
