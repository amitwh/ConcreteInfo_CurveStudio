"""Data layer: immutable datasets, reversible operations and validated fit inputs."""

from curvestudio.data.dataset import (
    XY,
    Column,
    Dataset,
    GroupColumn,
    coerce_numeric,
    group_labels,
)
from curvestudio.data.errors import DataError
from curvestudio.data.importers import ImportLimits, read_dataset, read_text
from curvestudio.data.operations import ExcludeRows, Operation, RenameColumn, TransformColumn
from curvestudio.data.validation import Issue, validate

__all__ = [
    "XY",
    "Column",
    "DataError",
    "Dataset",
    "ExcludeRows",
    "GroupColumn",
    "ImportLimits",
    "Issue",
    "Operation",
    "RenameColumn",
    "TransformColumn",
    "coerce_numeric",
    "group_labels",
    "read_dataset",
    "read_text",
    "validate",
]
