"""Data layer: immutable datasets, reversible operations and validated fit inputs."""

from curvestudio.data.dataset import XY, Dataset
from curvestudio.data.errors import DataError
from curvestudio.data.operations import ExcludeRows, Operation, RenameColumn, TransformColumn

__all__ = [
    "XY",
    "DataError",
    "Dataset",
    "ExcludeRows",
    "Operation",
    "RenameColumn",
    "TransformColumn",
]
