from __future__ import annotations as _

from collections.abc import Callable
from contextlib import contextmanager

import toolz

__all__ = [
    'smarter_curry',
    'force_curried',
]

_FORCE_CURRIED: bool = False

@contextmanager
def force_curried():
    """
    强制柯里化函数始终返回绑定函数，即便已经提供了所有必需的参数。见 :func:`smarter_curry` 中的实例

    Notes:
        - 此函数依赖没有被锁保护的全局变量，不是线程安全的
        - 如果在此环境中调用了没有被 `smarter_curry` 标注的函数，且其内部实现中以即时 (eager) 模式调用了被 `smarter_curry` 标注的函数，
        可能导致出现预期外的结果
    """
    global _FORCE_CURRIED
    old_value = _FORCE_CURRIED
    _FORCE_CURRIED = True
    try:
        yield
    finally:
        _FORCE_CURRIED = old_value

class _SmarterCurry(toolz.curry):
    def __call__(self, *args, **kwargs):
        if _FORCE_CURRIED:
            return self.bind(*args, **kwargs)
        return super().__call__(*args, **kwargs)

def smarter_curry[T, **P](func: Callable[P, T]) -> Callable[P, Callable[P, T] | T]:
    """
    对函数柯里化 `toolz.curry` 增强：
        - 在 `force_curried()` 上下文中，即使提供了所有参数也只会返回绑定函数，不会执行函数并返回结果。
          这在一些要构造无参数 `Callable` 的场景很方便
        - 增强类型，方便 IDE 分析类型与提示参数。
          注意受种种限制，当前写法不支持更新已传入的位置参数，因此对位置参数进行嵌套柯里化时无法获得正确的参数提示。尽请期待 Python 自身进化 🤷

    Notes:
        柯里化理论上可以作用于任意 `Callable`，不限于常规通过 `def` 定义的函数

    Examples:
        >>> @smarter_curry
        ... def add(a, b):
        ...     return a + b
        ...
        >>> # 常规柯里化行为：
        >>> #   - 提供部分参数时返回绑定函数
        >>> #   - 提供全部参数时执行函数并返回结果
        >>> add_one = add(1)
        >>> add_one(2)
        3
        >>> add(1, 2)
        3
        >>> with force_curried():
        ...     # 在 `force_curried` 上下文中，即便提供了所有参数也返回函数而非结果
        ...     bound_func = add(1, 2)
        ...
        >>> # 退出上下文后，函数可以被直接无传入参数调用并获得结果
        >>> bound_func()
        3
    """
    return _SmarterCurry(func)
