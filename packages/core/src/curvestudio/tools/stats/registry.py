"""The registry of the statistical tools: the one place that knows every tool.

A tool module builds a :class:`ToolSpec` and registers it when it is imported. The registry
imports the tool modules named in :data:`TOOL_MODULES` the first time it is used, so that no
import goes round in a circle: the package imports this module, and a tool module imports
the package.

The request side mirrors the result side. A surface never sends a column: it sends what a
person chose, column NAMES and options, and :meth:`Registry.build_request` turns the names
into the tool's typed request through the dataset's lookups. The builder reads each request
model's annotations to do that, so it has to know every field's shape: a field that names a
column in a form it cannot read is refused when the tool is registered.
"""

import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Annotated, Any, Literal, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticInvalidForJsonSchema,
    TypeAdapter,
    ValidationError,
)
from pydantic.fields import FieldInfo

from curvestudio.data import Column, DataError, GroupColumn
from curvestudio.tools._errors import ToolError
from curvestudio.tools._wording import listing
from curvestudio.tools.stats._base import ToolResult

FAMILIES: tuple[tuple[str, str], ...] = (
    ("summarise", "Summarise"),
    ("means", "Compare means"),
    ("ranks", "Rank-based comparisons"),
    ("relationships", "Relationships between columns"),
    ("counts", "Counts"),
    ("comparison", "Method comparison"),
    ("measurement", "Measurement and classification"),
    ("power", "Sample size and power"),
    ("advanced", "Advanced models"),
    ("evidence", "Evidence synthesis"),
    ("bayes", "Bayesian tests"),
)
"""The families a tool belongs to, each with its title, in the order a person meets them."""

_FAMILY_NAMES = frozenset(key for key, _ in FAMILIES)


class ToolSpec(BaseModel):
    """One tool: what it is called, what it takes and the function that runs it.

    ``max_rows`` is the most rows it is built for; the tool itself refuses more
    (``TOO_MANY_ROWS``). ``None`` when it has no such bound.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    family: str
    title: str
    request: type[BaseModel]
    run: Callable[[BaseModel], ToolResult]
    max_rows: int | None = None


FieldKind = Literal["column", "group", "columns", "option"]
"""How a request field is read: one column by name, one group column by name, several
columns by name, or an option passed on to the model to check."""


class RequestField(BaseModel):
    """One field of a request, as a surface shows it to a person.

    ``required`` is what the model says. ``option_schema`` is an option's JSON schema, built
    from the field's own type and ``Field`` (the whole model has none to give, since a column
    has none); ``None`` for the three column kinds, which are chosen by name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    label: str
    kind: FieldKind
    required: bool
    description: str | None
    option_schema: dict[str, Any] | None


class ToolInfo(BaseModel):
    """What a surface lists about one tool.

    ``needs_data``: the request has a field that names a column, so the tool runs on a
    dataset. ``fields`` come in the model's order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    family: str
    title: str
    needs_data: bool
    max_rows: int | None
    fields: tuple[RequestField, ...]


FIELD_LABELS: dict[str, str] = {
    "values": "Values",
    "a": "First column",
    "b": "Second column",
    "differences": "Differences",
    "groups": "Groups",
    "columns": "Columns",
    "outcome": "Outcome",
    "predictors": "Predictors",
    "measures": "Measurements",
    "rows": "Rows",
    "factor_a": "Factor A",
    "factor_b": "Factor B",
    "before": "Before",
    "after": "After",
    "method_a": "Method A",
    "method_b": "Method B",
}
"""The labels of the field names that more than one tool shares."""

ColumnReader = Callable[[str], Column]
GroupReader = Callable[[str], GroupColumn]

_UNIONS = (typing.Union, types.UnionType)
_SEVERAL = (Column, Ellipsis)
"""The arguments of ``tuple[Column, ...]``: the one way a field names several columns."""


NamedKind = Literal["column", "group", "columns"]
"""The three kinds of field whose value is a column name (or names) rather than an option."""


def _column_kind(annotation: Any) -> NamedKind | None:
    """The kind of an annotation the builder reads by name, or ``None`` for any other.

    Readable are ``Column``, ``GroupColumn`` and ``tuple[Column, ...]``, each optionally in a
    union with ``None``. Whether a field may be left empty is read from the field itself
    (``FieldInfo.is_required()``), not from here.
    """
    members = get_args(annotation) if get_origin(annotation) in _UNIONS else (annotation,)
    remaining = [member for member in members if member is not type(None)]
    if len(remaining) != 1:
        return None
    inner = remaining[0]
    if inner is Column:
        return "column"
    if inner is GroupColumn:
        return "group"
    if get_origin(inner) is tuple and get_args(inner) == _SEVERAL:
        return "columns"
    return None


def _names_a_column(annotation: Any) -> bool:
    """Whether ``Column`` or ``GroupColumn`` appears anywhere in ``annotation``: directly,
    inside a container, or inside a nested model's fields."""
    if annotation is Column or annotation is GroupColumn:
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return any(_names_a_column(field.annotation) for field in annotation.model_fields.values())
    return any(_names_a_column(argument) for argument in get_args(annotation))


def _option_schema(name: str, annotation: Any, info: FieldInfo) -> dict[str, Any]:
    """An option's JSON schema: the constraints, default and description of its field.

    The whole request model has no schema to give, since a column has none, so the option's
    is built from the field's own type and ``Field`` alone.

    Raises:
        ValueError: the option's type has no JSON schema.
    """
    try:
        return TypeAdapter(Annotated[annotation, info]).json_schema()
    except PydanticInvalidForJsonSchema as exc:
        raise ValueError(
            f"field {name!r} of the request model is an option with no JSON schema "
            f"({annotation}), so no form could show it"
        ) from exc


@dataclass(frozen=True)
class _Field:
    """One field of a request model, as the builder reads it and the catalogue describes it."""

    name: str
    info: FieldInfo
    kind: FieldKind
    option_schema: dict[str, Any] | None
    """An option's JSON schema; ``None`` for the three column kinds."""

    @classmethod
    def read(cls, name: str, info: FieldInfo) -> _Field:
        """Read one field of a request model as the model declares it.

        Raises:
            ValueError: the field names a column in a form the builder cannot read, or is an
                option with no JSON schema.
        """
        kind = _column_kind(info.annotation)
        if kind is not None:
            return cls(name, info, kind, None)
        if _names_a_column(info.annotation):
            raise ValueError(
                f"field {name!r} of the request model names a column in a form the builder "
                f"cannot read ({info.annotation}); it takes Column, GroupColumn or "
                "tuple[Column, ...]"
            )
        return cls(name, info, "option", _option_schema(name, info.annotation, info))

    def lookup(
        self, value: Any, column: ColumnReader, group: GroupReader
    ) -> Column | GroupColumn | tuple[Column, ...]:
        """The column or columns a person chose by name, read through the dataset's lookups.

        Raises:
            ToolError: ``INVALID_OPTION`` when ``value`` is not a name (or a list of names);
                the lookup's own code and message for a column the dataset cannot give.
        """
        if self.kind == "columns":
            if not isinstance(value, (list, tuple)):
                raise ToolError(
                    "INVALID_OPTION", f"{self.name} takes a list of column names, not {value!r}"
                )
            if not value:
                raise ToolError(
                    "INVALID_OPTION",
                    f"{self.name} takes at least one column name, not an empty list",
                )
            return tuple(self._one(name, column) for name in value)
        if self.kind == "group":
            return self._one(value, group)
        return self._one(value, column)

    def _one[ModelT: BaseModel](self, name: Any, reader: Callable[[str], ModelT]) -> ModelT:
        if not isinstance(name, str):
            raise ToolError("INVALID_OPTION", f"{self.name} takes a column name, not {name!r}")
        try:
            return reader(name)
        except DataError as exc:
            # A DataError keeps its message only as its first argument; ``str(exc)`` would add
            # the code and the rows to it.
            raise ToolError(exc.code, str(exc.args[0])) from exc

    def describe(self) -> RequestField:
        extra = self.info.json_schema_extra
        chosen = extra.get("label") if isinstance(extra, dict) else None
        return RequestField(
            name=self.name,
            label=str(chosen or FIELD_LABELS.get(self.name) or _label_from_name(self.name)),
            kind=self.kind,
            required=self.info.is_required(),
            description=self.info.description,
            option_schema=self.option_schema,
        )


def _label_from_name(name: str) -> str:
    """``factor`` reads ``Factor``; ``dose_per_day`` reads ``Dose per day``."""
    words = name.replace("_", " ")
    return words[:1].upper() + words[1:]


def _first_error(exc: ValidationError) -> str:
    """The first of a model's refusals, as ``field: message``."""
    first = exc.errors()[0]
    return f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}"


@dataclass(frozen=True)
class _Tool:
    """A registered spec with its request's fields, read once."""

    spec: ToolSpec
    fields: tuple[_Field, ...]


class Registry(Mapping[str, ToolSpec]):
    """The tools registered in it, by name, in the order they were registered.

    A private registry is made with ``Registry()``. The module-level :data:`TOOLS` is given
    :data:`TOOL_MODULES` and imports each of them the first time it is used.
    """

    def __init__(self, modules: Sequence[str] = ()) -> None:
        self._modules = tuple(modules)
        self._tools: dict[str, _Tool] = {}
        self._loaded = not self._modules

    def _load(self) -> None:
        """Import the tool modules once, the first time the registry is used.

        The flag is set before importing, so a module that looks the registry up as it is
        imported does not come back here; it is cleared again if an import fails, so the
        failure is raised again on the next use and never mistaken for an empty registry.
        """
        if self._loaded:
            return
        self._loaded = True
        try:
            for name in self._modules:
                import_module(name)
        except BaseException:
            self._loaded = False
            raise

    def __getitem__(self, name: str) -> ToolSpec:
        self._load()
        return self._tools[name].spec

    def __iter__(self) -> Iterator[str]:
        self._load()
        return iter(tuple(self._tools))

    def __len__(self) -> int:
        self._load()
        return len(self._tools)

    def register(self, spec: ToolSpec) -> ToolSpec:
        """Add a tool, and give the spec back.

        Raises:
            ValueError: a family that is not one of :data:`FAMILIES`; a name already
                registered; a field of the request model the builder cannot read or describe.
        """
        self._load()
        if spec.family not in _FAMILY_NAMES:
            raise ValueError(
                f"tool {spec.name!r} names the family {spec.family!r}, which is not one of the "
                f"families {', '.join(repr(title) for _, title in FAMILIES)}"
            )
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        fields = tuple(_Field.read(name, info) for name, info in spec.request.model_fields.items())
        self._tools[spec.name] = _Tool(spec, fields)
        return spec

    def _tool(self, name: str) -> _Tool:
        self._load()
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError("UNKNOWN_TOOL", f"there is no tool named {name!r}") from None

    def run(self, name: str, request: BaseModel) -> ToolResult:
        """Run the tool ``name`` on its own typed request.

        Raises:
            ToolError: ``UNKNOWN_TOOL``; ``INVALID_OPTION`` when the request is not the tool's
                own model. What the tool itself raises passes through unchanged.
        """
        spec = self._tool(name).spec
        if type(request) is not spec.request:
            raise ToolError(
                "INVALID_OPTION",
                f"{name} takes a {spec.request.__name__}, not a request of "
                f"{type(request).__name__}",
            )
        return spec.run(request)

    def build_request(
        self, tool: str, spec: Mapping[str, Any], *, column: ColumnReader, group: GroupReader
    ) -> BaseModel:
        """Turn what a person chose into the tool's typed request.

        Args:
            tool: the tool's name.
            spec: the person's choices by field name. A column field holds the column's NAME,
                or a list of names where several are taken; anything else is an option.
            column: the dataset's column lookup, by name.
            group: the dataset's group column lookup, by name.

        Raises:
            ToolError: ``UNKNOWN_TOOL``; ``INVALID_OPTION`` for a key that is not a field, a
                name that is not a string, or a value the model refuses (its first error's
                field and message); the lookup's own code and message for a column the
                dataset cannot give.
        """
        entry = self._tool(tool)
        known = {field.name for field in entry.fields}
        unknown = [key for key in spec if key not in known]
        if unknown:
            names = [field.name for field in entry.fields]
            takes = (
                f"{listing(names, 'field')} {'is' if len(names) == 1 else 'are'} all it takes"
                if names
                else "it takes no fields"
            )
            raise ToolError(
                "INVALID_OPTION",
                f"{tool} takes no {listing([repr(key) for key in unknown], 'option')}; {takes}",
            )
        values: dict[str, Any] = {}
        for field in entry.fields:
            if field.name not in spec:
                continue
            given = spec[field.name]
            if field.kind == "option":
                values[field.name] = given
            elif given is not None:
                values[field.name] = field.lookup(given, column, group)
            elif not field.info.is_required():
                values[field.name] = None
            # A required column field given nothing is left out: the model names it as missing.
        try:
            return entry.spec.request(**values)
        except ValidationError as exc:
            raise ToolError("INVALID_OPTION", _first_error(exc)) from exc

    def info(self, tool: str) -> ToolInfo:
        """What a surface lists about one tool.

        Raises:
            ToolError: ``UNKNOWN_TOOL``.
        """
        entry = self._tool(tool)
        return ToolInfo(
            name=entry.spec.name,
            family=entry.spec.family,
            title=entry.spec.title,
            needs_data=any(field.kind != "option" for field in entry.fields),
            max_rows=entry.spec.max_rows,
            fields=tuple(field.describe() for field in entry.fields),
        )

    def catalogue(self) -> tuple[ToolInfo, ...]:
        """Every tool, in the order they were registered."""
        return tuple(self.info(name) for name in self)


TOOL_MODULES: tuple[str, ...] = (
    "curvestudio.tools.stats.anova",
    "curvestudio.tools.stats.anova_more",
    "curvestudio.tools.stats.bayes",
    "curvestudio.tools.stats.comparison",
    "curvestudio.tools.stats.comparison_lines",
    "curvestudio.tools.stats.contingency",
    "curvestudio.tools.stats.correlation",
    "curvestudio.tools.stats.descriptives",
    "curvestudio.tools.stats.distribution",
    "curvestudio.tools.stats.factor",
    "curvestudio.tools.stats.logistic",
    "curvestudio.tools.stats.means",
    "curvestudio.tools.stats.meta",
    "curvestudio.tools.stats.mixed",
    "curvestudio.tools.stats.pca",
    "curvestudio.tools.stats.power",
    "curvestudio.tools.stats.quality",
    "curvestudio.tools.stats.ranks",
    "curvestudio.tools.stats.regression",
    "curvestudio.tools.stats.reliability",
    "curvestudio.tools.stats.roc",
    "curvestudio.tools.stats.survival",
)
"""A module per tool family, each registering its tool as it is imported."""

TOOLS = Registry(TOOL_MODULES)
"""The registry every surface reaches: the one the tool modules register in."""


def run_tool(name: str, request: BaseModel) -> ToolResult:
    """:meth:`Registry.run` on :data:`TOOLS`."""
    return TOOLS.run(name, request)


def build_request(
    tool: str, spec: Mapping[str, Any], *, column: ColumnReader, group: GroupReader
) -> BaseModel:
    """:meth:`Registry.build_request` on :data:`TOOLS`."""
    return TOOLS.build_request(tool, spec, column=column, group=group)


def catalogue() -> tuple[ToolInfo, ...]:
    """:meth:`Registry.catalogue` on :data:`TOOLS`."""
    return TOOLS.catalogue()
