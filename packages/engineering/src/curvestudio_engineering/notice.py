"""The engineering library's honesty notice and the factory that attaches it.

Every engineering model is an empirical or theoretical relationship with stated assumptions.
None is an acceptance equation of a code: a design code may be cited as the source of a form,
but using the fit to accept concrete still requires the code's own rules, sampling and
statistics. :func:`engineering_model` appends that sentence to every model it builds and
refuses a model that has no references, no example, or a category outside
``engineering.<sub>``, so the catalogue cannot drift from the notice.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter

FloatArray = NDArray[np.float64]

ENGINEERING_NOTICE = (
    "Empirical/theoretical relationship with the assumptions listed above; it is not an "
    "IS-code acceptance equation and must not be used as one."
)

ENGINEERING_SUBCATEGORIES = (
    "strength",
    "properties",
    "cementitious",
    "durability",
    "packing",
)
"""Allowed second segment of an ``engineering.*`` category."""


def engineering_model(
    *,
    id: str,
    name: str,
    category: str = "",
    expression: str,
    parameters: tuple[Parameter, ...],
    initial_guess: Callable[[FloatArray, FloatArray], dict[str, float]],
    variables: tuple[str, ...] = ("x",),
    domain_check: Callable[[FloatArray, FloatArray], list[str]] | None = None,
    references: tuple[str, ...] = (),
    assumptions: str = "",
    aliases: tuple[str, ...] = (),
    description: str = "",
    example: ModelExample | None = None,
) -> ModelSpec:
    """Build an engineering :class:`ModelSpec` with the honesty notice appended.

    The factory enforces the parts of the contract that are specific to the engineering
    library, so a model added later cannot quietly drop them:

    * the category is ``engineering.<sub>`` with ``<sub>`` one of
      :data:`ENGINEERING_SUBCATEGORIES`;
    * ``references`` is a non-empty tuple of non-empty strings (a citation that cannot be
      verified is omitted, never invented);
    * ``example`` is a :class:`ModelExample`, so the recovery harness has a truth to check.

    The notice :data:`ENGINEERING_NOTICE` is appended to ``assumptions`` once; passing
    assumptions that already end with it leaves them unchanged.

    Raises:
        ValueError: the category, references or example is missing or invalid.
    """
    head, _, tail = category.partition(".")
    if head != "engineering" or tail not in ENGINEERING_SUBCATEGORIES:
        raise ValueError(
            "engineering models need a category 'engineering.<sub>' with <sub> one of "
            f"{list(ENGINEERING_SUBCATEGORIES)}; got {category!r}"
        )
    if not isinstance(references, tuple) or not references:
        raise ValueError(
            "engineering models need non-empty references; omit a citation you cannot verify"
        )
    invalid = [
        reference
        for reference in references
        if not isinstance(reference, str) or not reference.strip()
    ]
    if invalid:
        raise ValueError(f"engineering references must be non-empty strings; got {invalid!r}")
    if not isinstance(example, ModelExample):
        raise ValueError(
            "engineering models need an example (a ModelExample) so the recovery harness can "
            "check them"
        )
    text = assumptions.strip()
    if not text.endswith(ENGINEERING_NOTICE):
        text = f"{text} {ENGINEERING_NOTICE}".strip()
    return ModelSpec(
        id=id,
        name=name,
        category=category,
        expression=expression,
        parameters=parameters,
        initial_guess=initial_guess,
        variables=variables,
        domain_check=domain_check,
        references=references,
        assumptions=text,
        aliases=aliases,
        description=description,
        example=example,
    )
