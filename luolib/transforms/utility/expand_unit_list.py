from monai import transforms as mt

__all__ = [
    'ExpandUnitList',
]

class ExpandUnitList(mt.Transform):
    def __call__(self, data):
        if isinstance(data, list) and len(data) == 1:
            return data[0]
