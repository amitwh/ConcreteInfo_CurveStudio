"""Reversible dataset operations.

Operations are immutable descriptions of a change. They are applied, in order, to a
copy of the original table by :class:`curvestudio.data.dataset.Dataset`.
"""

from dataclasses import dataclass
from typing import Literal

TransformFunc = Literal["log", "log10", "sqrt", "reciprocal", "square", "affine"]


@dataclass(frozen=True)
class ExcludeRows:
    """Exclude rows (0-based positions in the original table) from fitting."""

    rows: tuple[int, ...]


@dataclass(frozen=True)
class TransformColumn:
    """Write ``func(source)`` to column ``target``.

    ``affine`` computes ``scale * source + offset``; the other functions ignore
    ``scale`` and ``offset``. ``log`` is the natural logarithm.
    """

    source: str
    target: str
    func: TransformFunc
    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class RenameColumn:
    """Rename column ``old`` to ``new``."""

    old: str
    new: str


Operation = ExcludeRows | TransformColumn | RenameColumn
