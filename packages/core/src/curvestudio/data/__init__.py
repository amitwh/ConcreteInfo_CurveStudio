"""Data layer: immutable datasets, reversible operations and validated fit inputs."""

from curvestudio.data.dataset import XY, Dataset
from curvestudio.data.errors import DataError
from curvestudio.data.importers import ImportLimits, read_dataset, read_text
from curvestudio.data.operations import ExcludeRows, Operation, RenameColumn, TransformColumn
from curvestudio.data.validation import Issue, validate

__all__ = [
    "XY",
    "DataError",
    "Dataset",
    "ExcludeRows",
    "ImportLimits",
    "Issue",
    "Operation",
    "RenameColumn",
    "TransformColumn",
    "read_dataset",
    "read_text",
    "validate",
]
