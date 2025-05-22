from .device_map import *
from .einops import *
from .enums import DataSplit, DataKey
from .file_log import *
from .index_tracker import IndexTracker
from .misc import *
from .process_map import *
from .smarter_curry import *
from .zstd import *

class SimpleReprMixin(object):
    """A mixin implementing a simple __repr__."""
    def __repr__(self):
        return "<{cls} @{id:x} {attrs}>".format(
            cls=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )
