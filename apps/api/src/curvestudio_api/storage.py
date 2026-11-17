"""The stored form of a dataset: the cleaned x, y (and weight) columns as plain CSV bytes.

Floats are written with their shortest round-trip form, so reading the payload back gives
exactly the numbers that were fitted. There is no pickle and nothing from the upload
beyond these numbers is kept.
"""

import io

import numpy as np
import pandas as pd

from curvestudio.data import XY

_HEADER = ("x", "y", "w")


def encode_payload(xy: XY) -> bytes:
    """CSV bytes (header ``x,y`` or ``x,y,w``) for the data of ``xy``."""
    columns = {"x": xy.x, "y": xy.y}
    if xy.weights is not None:
        columns["w"] = xy.weights
    buffer = io.StringIO()
    pd.DataFrame(columns).to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("ascii")


def decode_payload(payload: bytes, *, x_name: str, y_name: str, fingerprint: str) -> XY:
    """The :class:`XY` stored by :func:`encode_payload`, with the original fingerprint."""
    header, _, body = payload.partition(b"\n")
    columns = tuple(header.decode("ascii").strip().split(","))
    if columns not in (_HEADER[:2], _HEADER):
        raise ValueError(f"stored dataset has unexpected columns {list(columns)}")
    values = np.loadtxt(
        io.BytesIO(body), delimiter=",", dtype=np.float64, ndmin=2, encoding="ascii"
    ).reshape(-1, len(columns))
    return XY(
        x=values[:, 0],
        y=values[:, 1],
        weights=values[:, 2] if len(columns) == 3 else None,
        x_name=x_name,
        y_name=y_name,
        fingerprint=fingerprint,
    )
