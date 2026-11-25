"""The registry of the statistical tools: the one place that knows every tool.

A tool registers a :class:`ToolSpec` when its module is imported. The registry turns what a
person chose (column names and options) into the tool's typed request, runs a tool on such
a request, and describes every tool for the surfaces that list them. These tests work on a
private registry with a toy request model; the tools themselves are tested where they live.
"""

from collections.abc import Mapping
from importlib import import_module
from typing import Literal

import numpy as np
import pytest
from pydantic import BaseModel, Field

from curvestudio.data import Column, DataError, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools.stats import (
    FAMILIES,
    Counts,
    Registry,
    ToolInfo,
    ToolResult,
    ToolSpec,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats import registry as registry_module
from curvestudio.tools.stats.registry import FIELD_LABELS, TOOL_MODULES, TOOLS


class ToyRequest(BaseModel):
    values: Column
    groups: GroupColumn | None = None
    more: tuple[Column, ...] | None = None
    level: float = Field(default=0.95, description="The confidence level.")
    method: Literal["a", "b"] = "a"


class OtherRequest(BaseModel):
    x: Column


class OptionsOnlyRequest(BaseModel):
    level: float = 0.95
    method: Literal["a", "b"] = "a"


class OddRequest(BaseModel):
    """A column field the builder has no way to read: a name would have to become a mapping."""

    mapping: dict[str, Column]


class LabelledRequest(BaseModel):
    factor_b: Column
    mixed_model: str = "linear"
    before: Column = Field(json_schema_extra={"label": "Earlier"})


RESULT = ToolResult(
    tool="toy",
    title="A toy comparison",
    level=0.95,
    interpretation="The two groups differ.",
    methods_fragment="A toy test was run.",
    counts=Counts(n_offered=3, n_used=3, n_excluded=0),
)
TOY_FIELDS = ("values", "groups", "more", "level", "method")


def _column(name: str, values: list[float]) -> Column:
    return Column(
        name=name,
        values=np.asarray(values, dtype=np.float64),
        rows=np.arange(len(values), dtype=np.intp),
    )


def _group(name: str, labels: list[str]) -> GroupColumn:
    return GroupColumn(
        name=name,
        labels=np.asarray(labels, dtype=object),
        rows=np.arange(len(labels), dtype=np.intp),
    )


def _spec(
    name: str = "toy",
    request: type[BaseModel] = ToyRequest,
    family: str = "means",
    run: object | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        family=family,
        title="A toy comparison",
        request=request,
        run=run if run is not None else (lambda _request: RESULT),  # type: ignore[arg-type]
    )


class Lookups:
    """What a dataset hands the builder: its columns and groups by name, with its refusals."""

    def __init__(self) -> None:
        self.columns = {
            "y": _column("y", [1.0, 2.0, 3.0]),
            "a": _column("a", [4.0, 5.0, 6.0]),
            "b": _column("b", [7.0, 8.0, 9.0]),
        }
        self.groups = {"g": _group("g", ["u", "v", "u"]), "h": _group("h", ["p", "q", "p"])}
        self.asked: list[tuple[str, str]] = []

    def column(self, name: str) -> Column:
        self.asked.append(("column", name))
        if name not in self.columns:
            raise DataError("UNKNOWN_COLUMN", f"Column {name!r} does not exist")
        return self.columns[name]

    def group(self, name: str) -> GroupColumn:
        self.asked.append(("group", name))
        if name not in self.groups:
            raise DataError("UNKNOWN_COLUMN", f"Column {name!r} does not exist")
        return self.groups[name]


def _registry() -> tuple[Registry, Lookups]:
    registry = Registry()
    registry.register(_spec())
    registry.register(_spec("other", OtherRequest, family="summarise"))
    return registry, Lookups()


def _build(registry: Registry, data: Lookups, tool: str, spec: dict[str, object]) -> BaseModel:
    return registry.build_request(tool, spec, column=data.column, group=data.group)


# --------------------------------------------------------------------------- building a request


def test_build_request_turns_column_names_into_the_typed_request() -> None:
    registry, data = _registry()

    request = _build(
        registry,
        data,
        "toy",
        {"values": "y", "groups": "g", "more": ["a", "b"], "level": 0.99, "method": "b"},
    )

    assert type(request) is ToyRequest
    assert request.values is data.columns["y"]
    assert request.groups is data.groups["g"]
    assert tuple(column.name for column in request.more or ()) == ("a", "b")
    assert request.level == 0.99 and request.method == "b"
    assert data.asked == [("column", "y"), ("group", "g"), ("column", "a"), ("column", "b")]


def test_build_request_leaves_an_optional_column_empty_and_options_at_their_defaults() -> None:
    registry, data = _registry()

    bare = _build(registry, data, "toy", {"values": "y"})
    explicit = _build(registry, data, "toy", {"values": "y", "groups": None, "more": None})

    assert bare.groups is None and bare.more is None
    assert bare.level == 0.95 and bare.method == "a"
    assert explicit == bare
    # Each build read one name, and no group: nothing was asked for the fields left empty.
    assert data.asked == [("column", "y")] * 2


def test_build_request_passes_an_option_through_unchanged() -> None:
    registry, data = _registry()
    request = _build(registry, data, "toy", {"values": "y", "level": 0.9})
    assert isinstance(request, ToyRequest) and request.level == 0.9


@pytest.mark.parametrize(
    ("spec", "code", "named"),
    [
        ({"values": "y", "extra": 1}, "INVALID_OPTION", "extra"),
        ({"values": 3}, "INVALID_OPTION", "values"),
        ({"values": ["y"]}, "INVALID_OPTION", "values"),
        ({"values": "y", "groups": ["g"]}, "INVALID_OPTION", "groups"),
        ({"values": "y", "groups": 0}, "INVALID_OPTION", "groups"),
        ({"values": "y", "more": ["a", 2]}, "INVALID_OPTION", "more"),
        ({"values": "y", "more": []}, "INVALID_OPTION", "more"),
        ({"values": "y", "more": "a"}, "INVALID_OPTION", "more"),
        ({"values": "nope"}, "UNKNOWN_COLUMN", "nope"),
        ({"values": "y", "groups": "nope"}, "UNKNOWN_COLUMN", "nope"),
        ({"values": "y", "more": ["a", "nope"]}, "UNKNOWN_COLUMN", "nope"),
        ({"values": "y", "method": "c"}, "INVALID_OPTION", "method"),
        ({"values": "y", "method": 1}, "INVALID_OPTION", "method"),
        ({"level": 0.9}, "INVALID_OPTION", "values"),
        ({"values": None}, "INVALID_OPTION", "values"),
    ],
)
def test_build_request_refuses_what_a_person_cannot_choose(
    spec: dict[str, object], code: str, named: str
) -> None:
    registry, data = _registry()
    with pytest.raises(ToolError) as refused:
        _build(registry, data, "toy", spec)
    assert refused.value.code == code
    assert named in refused.value.message


def test_an_unknown_option_names_the_tool_and_the_fields_it_does_take() -> None:
    registry, data = _registry()
    with pytest.raises(ToolError) as refused:
        _build(registry, data, "toy", {"values": "y", "alternative": "greater"})
    assert refused.value.code == "INVALID_OPTION"
    assert "alternative" in refused.value.message and "toy" in refused.value.message
    for field in TOY_FIELDS:
        assert field in refused.value.message


def test_a_missing_column_keeps_the_data_layer_s_code_and_message() -> None:
    registry, data = _registry()
    with pytest.raises(ToolError) as refused:
        _build(registry, data, "toy", {"values": "nope"})
    assert refused.value.code == "UNKNOWN_COLUMN"
    assert refused.value.message == "Column 'nope' does not exist"
    assert str(refused.value) == "UNKNOWN_COLUMN: Column 'nope' does not exist"


def test_a_tool_error_raised_by_a_lookup_passes_through() -> None:
    registry, data = _registry()
    refusal = ToolError("NON_NUMERIC", "Column 'd' holds dates, not numbers")

    def column(name: str) -> Column:
        raise refusal

    with pytest.raises(ToolError) as refused:
        registry.build_request("toy", {"values": "d"}, column=column, group=data.group)
    assert refused.value is refusal


def test_a_failed_check_of_the_model_is_an_option_error_naming_the_field() -> None:
    registry, data = _registry()
    with pytest.raises(ToolError) as refused:
        _build(registry, data, "toy", {"values": "y", "method": "c"})
    assert refused.value.code == "INVALID_OPTION"
    assert refused.value.message.startswith("method: ")

    with pytest.raises(ToolError) as missing:
        _build(registry, data, "toy", {"level": 0.9})
    assert missing.value.message == "values: Field required"


def test_an_unknown_tool_is_refused_by_every_entry_point() -> None:
    registry, data = _registry()
    for attempt in (
        lambda: _build(registry, data, "no-such-tool", {"values": "y"}),
        lambda: registry.info("no-such-tool"),
        lambda: registry.run("no-such-tool", ToyRequest(values=data.columns["y"])),
    ):
        with pytest.raises(ToolError) as refused:
            attempt()
        assert refused.value.code == "UNKNOWN_TOOL"
        assert "no-such-tool" in refused.value.message


# --------------------------------------------------------------------------- running a tool


def test_run_passes_the_request_to_the_tool_and_returns_its_result() -> None:
    seen: list[BaseModel] = []

    def run(request: BaseModel) -> ToolResult:
        seen.append(request)
        return RESULT

    registry = Registry()
    registry.register(_spec(run=run))
    request = ToyRequest(values=_column("y", [1.0]))
    assert registry.run("toy", request) is RESULT
    assert seen == [request]


def test_run_refuses_a_request_that_is_not_the_tools_own_model() -> None:
    registry, data = _registry()
    foreign = OtherRequest(x=data.columns["y"])
    with pytest.raises(ToolError) as refused:
        registry.run("toy", foreign)
    assert refused.value.code == "INVALID_OPTION"
    assert "ToyRequest" in refused.value.message and "OtherRequest" in refused.value.message


def test_run_passes_the_tool_s_own_error_through() -> None:
    refusal = ToolError("TOO_FEW_ROWS", "the toy needs at least two rows")

    def run(request: BaseModel) -> ToolResult:
        raise refusal

    registry = Registry()
    registry.register(_spec(run=run))
    with pytest.raises(ToolError) as refused:
        registry.run("toy", ToyRequest(values=_column("y", [1.0])))
    assert refused.value is refusal


# --------------------------------------------------------------------------- describing the tools


def test_info_lists_every_field_with_its_kind_label_and_requirement() -> None:
    registry, _ = _registry()
    info = registry.info("toy")
    assert isinstance(info, ToolInfo)
    assert (info.name, info.family, info.title, info.needs_data, info.max_rows) == (
        "toy",
        "means",
        "A toy comparison",
        True,
        None,
    )
    # "more" is in no list of known labels, so it reads as its own name with a capital letter.
    assert [(field.name, field.kind, field.label, field.required) for field in info.fields] == [
        ("values", "column", "Values", True),
        ("groups", "group", "Groups", False),
        ("more", "columns", "More", False),
        ("level", "option", "Level", False),
        ("method", "option", "Method", False),
    ]
    assert [field.option_schema is None for field in info.fields] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert info.fields[3].description == "The confidence level."
    assert info.fields[4].description is None


def test_an_option_carries_the_schema_its_field_would_publish() -> None:
    registry, _ = _registry()
    level, method = registry.info("toy").fields[3:]

    assert level.option_schema is not None
    assert level.option_schema["type"] == "number"
    assert level.option_schema["description"] == "The confidence level."
    assert level.option_schema["default"] == 0.95
    assert method.option_schema is not None
    assert method.option_schema["type"] == "string"
    assert method.option_schema["enum"] == ["a", "b"]


def test_a_constrained_option_and_an_enum_option_publish_their_rules() -> None:
    class StrictRequest(BaseModel):
        n: int = Field(ge=2, le=50, description="Rows per group.")
        alternative: Literal["two-sided", "greater"] = "two-sided"
        ratio: float | None = None

    registry = Registry()
    registry.register(_spec("strict", StrictRequest, family="power"))
    info = registry.info("strict")
    assert info.needs_data is False
    rows, alternative, ratio = info.fields
    assert rows.option_schema is not None
    assert (
        rows.option_schema["type"],
        rows.option_schema["minimum"],
        rows.option_schema["maximum"],
    ) == (
        "integer",
        2,
        50,
    )
    assert rows.option_schema["description"] == "Rows per group."
    assert alternative.option_schema is not None
    assert alternative.option_schema["enum"] == ["two-sided", "greater"]
    assert ratio.option_schema is not None
    assert {"type": "number"} in ratio.option_schema["anyOf"] and {
        "type": "null"
    } in ratio.option_schema["anyOf"]


def test_a_label_comes_from_the_field_then_the_known_names_then_its_own_name() -> None:
    registry = Registry()
    registry.register(_spec("labelled", LabelledRequest, family="summarise"))
    fields = registry.info("labelled").fields
    assert [(field.name, field.label, field.kind) for field in fields] == [
        ("factor_b", "Factor B", "column"),
        ("mixed_model", "Mixed model", "option"),
        ("before", "Earlier", "column"),
    ]
    assert FIELD_LABELS["method_a"] == "Method A" and FIELD_LABELS["values"] == "Values"


def test_an_options_only_tool_needs_no_data() -> None:
    registry = Registry()
    registry.register(_spec("options", OptionsOnlyRequest, family="power"))
    assert registry.info("options").needs_data is False
    assert [field.kind for field in registry.info("options").fields] == ["option", "option"]


def test_catalogue_lists_the_tools_in_the_order_they_were_registered() -> None:
    registry, _ = _registry()
    registry.register(_spec("third", OptionsOnlyRequest, family="counts", run=lambda r: RESULT))
    assert [info.name for info in registry.catalogue()] == ["toy", "other", "third"]
    assert registry.catalogue()[0] == registry.info("toy")
    assert list(registry) == ["toy", "other", "third"]
    assert len(registry) == 3 and "toy" in registry and "none" not in registry
    assert registry["toy"].request is ToyRequest
    with pytest.raises(KeyError):
        registry["none"]


def test_max_rows_is_published() -> None:
    registry = Registry()
    registry.register(
        ToolSpec(
            name="capped",
            family="means",
            title="Capped",
            request=ToyRequest,
            run=lambda r: RESULT,
            max_rows=10_000,
        )
    )
    assert registry.info("capped").max_rows == 10_000


# --------------------------------------------------------------------------- registering


def test_register_refuses_a_field_it_cannot_read_and_an_unknown_family() -> None:
    registry = Registry()
    with pytest.raises(ValueError, match="mapping"):
        registry.register(_spec("odd", OddRequest))
    with pytest.raises(ValueError, match="tarot"):
        registry.register(_spec("fortune", family="tarot"))
    assert list(registry) == []

    # Nothing of a refused registration is left behind, and a good one follows cleanly.
    registry.register(_spec())
    assert list(registry) == ["toy"]
    with pytest.raises(ValueError, match="toy"):
        registry.register(_spec())


def test_register_gives_back_what_it_was_given() -> None:
    registry = Registry()
    spec = _spec()
    assert registry.register(spec) is spec


def test_a_registry_imports_its_tool_modules_once_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    registry: Registry

    def import_module(name: str) -> object:
        """What importing a tool module does: it registers its spec, here on this registry."""
        imported.append(name)
        registry.register(_spec(f"tool{len(imported)}"))
        return object()

    monkeypatch.setattr(registry_module, "import_module", import_module)
    registry = Registry(["tools.one", "tools.two"])
    assert imported == []
    assert list(registry) == ["tool1", "tool2"]
    assert imported == ["tools.one", "tools.two"]
    registry.catalogue()
    registry.run("tool2", ToyRequest(values=_column("y", [1.0])))
    assert len(imported) == 2


# --------------------------------------------------------------------------- the module level


def test_the_module_level_registry_is_the_one_every_surface_reaches() -> None:
    assert isinstance(TOOLS, Registry) and isinstance(TOOLS, Mapping)
    assert isinstance(catalogue(), tuple)
    assert all(isinstance(info, ToolInfo) for info in catalogue())
    with pytest.raises(ToolError) as refused:
        run_tool("no-such-tool", ToyRequest(values=_column("y", [1.0])))
    assert refused.value.code == "UNKNOWN_TOOL"
    with pytest.raises(ToolError) as refused:
        build_request("no-such-tool", {}, column=Lookups().column, group=Lookups().group)
    assert refused.value.code == "UNKNOWN_TOOL"
    # The tuple grows by one module with every family, so it is not pinned to a list here.
    # What must hold: every name is a real module, and every module registered its tools.
    for module_name in TOOL_MODULES:
        assert import_module(module_name).__name__ == module_name
    assert len(catalogue()) >= len(TOOL_MODULES)


def test_the_families_are_named_in_the_order_a_person_meets_them() -> None:
    assert FAMILIES == (
        ("summarise", "Summarise"),
        ("means", "Compare means"),
        ("ranks", "Rank-based comparisons"),
        ("relationships", "Relationships between columns"),
        ("counts", "Counts"),
        ("comparison", "Method comparison"),
        ("measurement", "Measurement and classification"),
        ("power", "Sample size and power"),
        ("evidence", "Evidence synthesis"),
    )
