"""Early validation of a :class:`FitSpec`, so a bad equation is a 422 and not a failed job."""

import numpy as np

from curvestudio.data import XY
from curvestudio.workflow import FitSpec, fit_dataset

DRY_RUN_POINTS = 12
DRY_RUN_TIME_LIMIT_S = 1.0


def check_spec(spec: FitSpec, *, weighted: bool) -> None:
    """Run ``spec`` once on a small synthetic data set to surface configuration mistakes.

    A fit's *numerical* outcome never raises in the workflow layer, so only a specification
    the engine cannot run (bad equation, unknown model, impossible method/bounds, weights
    asked for but absent) gets through as a :class:`WorkflowError`, exactly as it would on
    the real data. ``weighted`` says whether the real data has a weights column.

    Raises:
        WorkflowError: the workflow layer's own error, with its code and hint.
    """
    x = np.linspace(1.0, 10.0, DRY_RUN_POINTS)
    xy = XY(
        x=x,
        y=2.0 * x + 1.0 + 0.1 * np.sin(7.0 * x),
        weights=np.ones_like(x) if weighted else None,
        x_name="x",
        y_name="y",
        fingerprint="dry-run",
    )
    probe = spec.model_copy(
        update={"time_limit": min(spec.time_limit or DRY_RUN_TIME_LIMIT_S, DRY_RUN_TIME_LIMIT_S)}
    )
    fit_dataset(xy, probe, compact=True)
