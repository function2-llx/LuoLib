from collections.abc import Iterable

import pandas as pd
from pandas._typing import DropKeep

__all__ = [
    'concat_drop_dup',
]


def concat_drop_dup(
    objs: Iterable[pd.Series | pd.DataFrame],
    keep: DropKeep = 'last',
) -> pd.DataFrame | pd.Series:
    ret = pd.concat(objs)
    return ret.loc[~ret.index.duplicated(keep)]
