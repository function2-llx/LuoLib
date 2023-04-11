from monai.utils import StrEnum

class DataSplit(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class DataKey(StrEnum):
    CASE = 'case'
    IMG = 'img'
    CLS = 'cls'
    SEG = 'seg'
    SEG_ORIGIN = 'seg-origin'
    CLINICAL = 'clinical'
