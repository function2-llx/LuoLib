from collections.abc import Sequence

from monai.data import list_data_collate as list_data_collate_monai

def list_data_collate(batch: Sequence):
    data = [
        x
        for item in batch
        for x in (item if isinstance(item, list) else [item])
    ]
    return list_data_collate_monai(data)
