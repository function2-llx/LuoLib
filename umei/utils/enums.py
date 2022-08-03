from monai.utils import StrEnum

class DataSplit(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class DataKey(StrEnum):
    IMG = 'img'
    SEG = 'seg'
    CLS = 'cls'
