"""Which models Auto-fit will consider for a run (no fitting here)."""

import logging

from curvestudio.discovery.types import (
    AutoFitConfig,
    CandidateRef,
    PolynomialRef,
    RegisteredRef,
)
from curvestudio.models import get_model, list_model_ids, list_models

_log = logging.getLogger(__name__)

MIN_RESIDUAL_DF = 3
"""A candidate needs at least ``k + 3`` active points."""

DEFAULT_MAX_PARAMS = 6
EXHAUSTIVE_MAX_PARAMS = 10


def parameter_cap(config: AutoFitConfig, n_active: int) -> int:
    """Largest parameter count to try: the configured cap, never above ``n_active - 3``."""
    if config.max_params is not None:
        cap = config.max_params
    elif config.exhaustive:
        cap = EXHAUSTIVE_MAX_PARAMS
    else:
        cap = min(DEFAULT_MAX_PARAMS, max(1, n_active // 4))
    return min(cap, n_active - MIN_RESIDUAL_DF)


def parameter_count(ref: CandidateRef) -> int:
    """Number of fitted parameters of ``ref``."""
    if isinstance(ref, PolynomialRef):
        return ref.parameter_count
    return len(get_model(ref.model_id).parameters)


def category_matches(category: str, families: tuple[str, ...] | None) -> bool:
    """``category`` equals a family or has it as first dotted segment; ``None`` keeps all."""
    if families is None:
        return True
    return any(category == f or category.startswith(f + ".") for f in families)


def enumerate_candidates(config: AutoFitConfig, n_active: int) -> list[CandidateRef]:
    """Candidates for ``n_active`` usable points, ordered by id.

    Polynomials start at degree 1 (``poly1`` is the straight line); a polynomial through the
    origin of degree ``d`` has ``d`` parameters, a free one ``d + 1``. Registered models are
    kept when their parameter count is within :func:`parameter_cap` and their category
    passes ``config.families``. Models that need a lazily generated family are dropped when
    ``include_generated`` is false. A family member that cannot be built is logged and
    left out; it cannot be fitted.
    """
    cap = parameter_cap(config, n_active)
    refs: list[CandidateRef] = []
    if cap < 1:
        return refs

    if config.include_polynomials and category_matches("polynomial", config.families):
        for degree in range(1, cap + 1):
            refs.append(PolynomialRef(degree=degree, through_origin=True))
            if degree + 1 <= cap:
                refs.append(PolynomialRef(degree=degree, through_origin=False))

    explicit = {m.id for m in list_models()}
    for model_id in list_model_ids(exhaustive=config.exhaustive):
        if not config.include_generated and model_id not in explicit:
            continue
        try:
            spec = get_model(model_id)
        except Exception as exc:  # a broken family member must not stop the whole run
            _log.warning("model %r skipped: %s: %s", model_id, type(exc).__name__, exc)
            continue
        if len(spec.parameters) <= cap and category_matches(spec.category, config.families):
            refs.append(RegisteredRef(model_id=model_id))

    return sorted(refs, key=lambda ref: ref.id)
