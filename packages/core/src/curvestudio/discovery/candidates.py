"""Which models Auto-fit will consider for a run (no fitting here)."""

import logging
import re
from functools import lru_cache

from curvestudio.discovery.types import (
    AutoFitConfig,
    AutoFitError,
    CandidateRef,
    CustomRef,
    PolynomialRef,
    RegisteredRef,
)
from curvestudio.models import (
    CustomModelDefinition,
    ModelSpec,
    build_custom_model,
    get_model,
    list_model_ids,
    list_models,
)

_log = logging.getLogger(__name__)

MIN_RESIDUAL_DF = 3
"""A candidate needs at least ``k + 3`` active points."""

DEFAULT_MAX_PARAMS = 6
EXHAUSTIVE_MAX_PARAMS = 10


def parameter_cap(config: AutoFitConfig, n_active: int) -> int:
    """Largest parameter count to try, never above ``n_active - 3``.

    ``config.max_params`` when given; otherwise ``n_active - 3`` when ``config.families`` is
    explicit (the user asked for those models: only the data can rule one out), otherwise
    ``min(10, n_active - 3)`` when exhaustive, else ``min(6, max(2, n_active // 3))``: a model
    gets three points per parameter before it is tried by default, with a floor of two.
    """
    if config.max_params is not None:
        cap = config.max_params
    elif config.families is not None:
        cap = n_active - MIN_RESIDUAL_DF
    elif config.exhaustive:
        cap = EXHAUSTIVE_MAX_PARAMS
    else:
        cap = min(DEFAULT_MAX_PARAMS, max(2, n_active // 3))
    return min(cap, n_active - MIN_RESIDUAL_DF)


@lru_cache(maxsize=256)
def _build_custom(definition: CustomModelDefinition) -> ModelSpec:
    return build_custom_model(definition)


def resolve_spec(ref: CandidateRef) -> ModelSpec | None:
    """The model behind ``ref``; ``None`` for a polynomial (those are fitted in closed form).

    A custom definition is compiled once per process and then reused.

    Raises:
        KeyError: a registered id that is unknown.
        ValueError: a custom definition that cannot be built.
    """
    if isinstance(ref, PolynomialRef):
        return None
    if isinstance(ref, CustomRef):
        return _build_custom(ref.definition)
    return get_model(ref.model_id)


def candidate_ref(config: AutoFitConfig, candidate_id: str) -> CandidateRef:
    """The reference for an id from a result: a polynomial, a custom model or a registered one.

    Raises:
        KeyError: the id is none of those (the message suggests close registered ids).
    """
    poly = _POLYNOMIAL_ID.fullmatch(candidate_id)
    if poly is not None:
        degree = int(candidate_id[len("poly") :].split("_")[0])
        if degree >= 1:
            return PolynomialRef(degree=degree, through_origin=candidate_id.endswith("_origin"))
    for definition in config.custom_models:
        if definition.id == candidate_id:
            return CustomRef(definition=definition)
    get_model(candidate_id)  # raises KeyError with suggestions for an unknown id
    return RegisteredRef(model_id=candidate_id)


def parameter_count(ref: CandidateRef) -> int:
    """Number of fitted parameters of ``ref``."""
    if isinstance(ref, PolynomialRef):
        return ref.parameter_count
    spec = resolve_spec(ref)
    assert spec is not None
    return len(spec.parameters)


def category_matches(category: str, families: tuple[str, ...] | None) -> bool:
    """``category`` equals a family or has it as first dotted segment; ``None`` keeps all."""
    if families is None:
        return True
    return any(category == f or category.startswith(f + ".") for f in families)


def enumerate_candidates(config: AutoFitConfig, n_active: int) -> list[CandidateRef]:
    """Candidates for ``n_active`` usable points, ordered by id.

    See :func:`enumerate_with_capped`; this is its first result.
    """
    return enumerate_with_capped(config, n_active)[0]


def enumerate_with_capped(
    config: AutoFitConfig, n_active: int
) -> tuple[list[CandidateRef], dict[str, int]]:
    """Candidates for ``n_active`` usable points, ordered by id, and those the cap left out.

    Polynomials start at degree 1 (``poly1`` is the straight line); a polynomial through the
    origin of degree ``d`` has ``d`` parameters, a free one ``d + 1``. Registered models are
    kept when their parameter count is within :func:`parameter_cap` and their category
    passes ``config.families``. Models that need a lazily generated family are dropped when
    ``include_generated`` is false. A family member that cannot be built is logged and
    left out; it cannot be fitted. Custom models of the configuration are always included
    (the data decides in :func:`~curvestudio.discovery.screen` whether one can be fitted).

    The second result maps the id of every registered model that passes the category filter
    but has more parameters than the cap to its parameter count, plus the smallest over-cap
    polynomials (the origin and free members one degree above the cap). Free polynomial
    degrees are unbounded, so only that next step is listed; the caller can report every
    entry instead of hiding it.

    Raises:
        AutoFitError: ``INVALID_CUSTOM_MODEL`` when a custom definition cannot be built,
            ``DUPLICATE_MODEL_ID`` when custom ids repeat or collide with another candidate.
    """
    cap = parameter_cap(config, n_active)
    refs: list[CandidateRef] = []
    capped: dict[str, int] = {}

    if config.include_polynomials and category_matches("polynomial", config.families):
        for degree in range(1, cap + 2):
            for through_origin in (True, False):
                ref = PolynomialRef(degree=degree, through_origin=through_origin)
                if ref.parameter_count <= cap:
                    refs.append(ref)
                elif ref.parameter_count == cap + 1:  # the next step up; higher ones are unbounded
                    capped[ref.id] = ref.parameter_count

    explicit = {m.id for m in list_models()}
    for model_id in list_model_ids(exhaustive=config.exhaustive):
        if not config.include_generated and model_id not in explicit:
            continue
        try:
            spec = get_model(model_id)
        except Exception as exc:  # a broken family member must not stop the whole run
            _log.warning("model %r skipped: %s: %s", model_id, type(exc).__name__, exc)
            continue
        if not category_matches(spec.category, config.families):
            continue
        if len(spec.parameters) <= cap:
            refs.append(RegisteredRef(model_id=model_id))
        else:
            capped[model_id] = len(spec.parameters)

    _add_custom(refs, config)
    return sorted(refs, key=lambda ref: ref.id), capped


_POLYNOMIAL_ID = re.compile(r"poly\d+(_origin)?")


def _add_custom(refs: list[CandidateRef], config: AutoFitConfig) -> None:
    """Append the configuration's custom models, refusing unusable or clashing ones."""
    if not config.custom_models:
        return
    taken = {ref.id for ref in refs} | set(list_model_ids(exhaustive=True))
    for definition in config.custom_models:
        ref = CustomRef(definition=definition)
        try:
            resolve_spec(ref)
        except ValueError as exc:
            raise AutoFitError(
                "INVALID_CUSTOM_MODEL", f"custom model {definition.id!r}: {exc}"
            ) from exc
        if definition.id in taken or _POLYNOMIAL_ID.fullmatch(definition.id):
            raise AutoFitError(
                "DUPLICATE_MODEL_ID",
                f"custom model id {definition.id!r} is already used by another candidate; "
                "give it a unique id",
            )
        taken.add(definition.id)
        refs.append(ref)
