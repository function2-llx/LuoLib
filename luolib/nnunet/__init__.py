from pathlib import Path

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results

__all__ = [
    'nnUNet_raw',
    'nnUNet_preprocessed',
    'nnUNet_results',
]

nnUNet_raw = Path(nnUNet_raw)
nnUNet_preprocessed = Path(nnUNet_preprocessed)
nnUNet_results = Path(nnUNet_results)
