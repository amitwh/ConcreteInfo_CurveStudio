"""SelectionController: which dataset is active and which columns it is shown by."""

import pandas as pd
import pytest
from desktop_data import GOOD_CSV

from curvestudio.data import Dataset
from curvestudio.project import Project, ProjectDataset
from curvestudio_desktop.controllers.selection import (
    SelectionController,
    default_columns,
)

CREATED = "2026-11-17T10:00:00+05:30"


def direct_project(frame: dict, name: str = "words") -> Project:
    """A project built directly (import would refuse <2 numeric columns)."""
    dataset = Dataset(name, pd.DataFrame(frame))
    return Project(
        name="Direct",
        created_at=CREATED,
        datasets=(ProjectDataset(id=name, name=name, dataset=dataset),),
    )


@pytest.fixture()
def selection() -> SelectionController:
    return SelectionController()


def make_project(controller, write_file, name: str, text: str):
    assert controller.import_dataset_file(write_file(name, text))
    return controller.project


def test_defaults_pick_first_two_numeric_columns(write_file, controller) -> None:
    text = "id,x,y,note\nd1,1.0,2.0,a\nd2,2.0,4.1,b\nd3,3.0,6.2,c\nd4,4.0,7.9,d\nd5,5.0,10.3,e\n"
    make_project(controller, write_file, "points.csv", text)
    dataset = controller.project.datasets[0].dataset
    assert default_columns(dataset) == ("x", "y")


def test_defaults_return_none_without_two_numeric_columns() -> None:
    project = direct_project({"a": ["one", "two", "three"], "b": ["x", "y", "z"]})
    assert default_columns(project.datasets[0].dataset) is None


def test_active_follows_project(controller, selection, write_file) -> None:
    project = make_project(controller, write_file, "points.csv", GOOD_CSV)
    selection.set_project(project)
    assert selection.active_id == project.datasets[0].id
    from curvestudio_desktop.controllers.providers import UnsavedAnswer

    controller.confirmation.answers.append(UnsavedAnswer.DISCARD)
    assert controller.new_project() is True
    selection.set_project(controller.project)
    assert selection.active_id is None


def test_active_survives_refresh_when_still_present(controller, selection, write_file) -> None:
    project = make_project(controller, write_file, "points.csv", GOOD_CSV)
    selection.set_project(project)
    keep_id = project.datasets[0].id
    controller.import_dataset_file(write_file("more.csv", GOOD_CSV))
    selection.set_project(controller.project)
    assert selection.active_id == keep_id


def test_columns_stored_and_xy_extraction(controller, selection, write_file) -> None:
    project = make_project(controller, write_file, "points.csv", GOOD_CSV)
    entry_id = project.datasets[0].id
    selection.set_project(project)
    assert selection.columns(entry_id) == ("x", "y", None)
    selection.set_issues(entry_id, ())
    xy = selection.xy()
    assert xy is not None
    assert xy.x_name == "x" and xy.y_name == "y"
    assert len(xy.x) == 5


def test_xy_is_none_without_usable_columns(selection) -> None:
    project = direct_project(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.1, 6.2, 7.9, 10.3],
            "z": ["a", "b", "c", "d", "e"],
        }
    )
    selection.set_project(project)
    selection.set_columns("words", "z", "y", None)  # z is text
    assert selection.xy() is None
    assert selection.last_error


def test_set_project_signals_only_when_active_changes(controller, selection, write_file) -> None:
    project = make_project(controller, write_file, "points.csv", GOOD_CSV)
    emits: list[bool] = []
    selection.active_changed.connect(lambda: emits.append(True))
    selection.set_project(project)  # first sight: None -> first dataset
    assert selection.active_id == project.datasets[0].id
    selection.set_project(project)  # same project again: nothing changed
    controller.import_dataset_file(write_file("more.csv", GOOD_CSV))
    selection.set_project(controller.project)  # active survives: still no change
    assert emits == [True]


def test_issues_round_trip(controller, selection, write_file) -> None:
    from curvestudio.workflow import ImportIssue

    project = make_project(controller, write_file, "points.csv", GOOD_CSV)
    entry_id = project.datasets[0].id
    selection.set_project(project)
    issue = ImportIssue(code="X", severity="warning", message="m", rows=(1,))
    selection.set_issues(entry_id, [issue])
    assert selection.issues(entry_id) == (issue,)


def _project_with_desktop_state(frame: dict, stored: dict) -> Project:
    dataset = Dataset("words", pd.DataFrame(frame))
    return Project(
        name="Stored",
        created_at=CREATED,
        datasets=(ProjectDataset(id="words", name="words", dataset=dataset),),
        plot_settings={"desktop": {"datasets": {"words": stored}}},
    )


NUMERIC_FRAME = {
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [2.0, 4.1, 6.2, 7.9, 10.3],
    "w": [1.0, 1.5, 1.0, 2.0, 1.0],
}


def test_set_project_restores_persisted_columns_and_issues(selection) -> None:
    from curvestudio.workflow import ImportIssue

    stored = {
        "columns": ["x", "y", "w"],
        "issues": [{"code": "DUPLICATE_ROW", "severity": "warning", "message": "m", "rows": [1]}],
    }
    selection.set_project(_project_with_desktop_state(NUMERIC_FRAME, stored))
    assert selection.active_id == "words"
    assert selection.columns("words") == ("x", "y", "w")
    assert selection.issues("words") == (
        ImportIssue(code="DUPLICATE_ROW", severity="warning", message="m", rows=(1,)),
    )
    xy = selection.xy()
    assert xy is not None and xy.weights is not None


def test_set_project_ignores_columns_that_no_longer_exist(selection) -> None:
    stored = {"columns": ["x", "gone", "w"], "issues": []}
    selection.set_project(_project_with_desktop_state(NUMERIC_FRAME, stored))
    assert selection.columns("words") == ("x", "y", None)  # falls back to the default


def test_set_project_tolerates_a_corrupt_desktop_blob(selection) -> None:
    stored = {"columns": "not-a-list", "issues": [{"severity": "bogus"}]}
    selection.set_project(_project_with_desktop_state(NUMERIC_FRAME, stored))
    assert selection.columns("words") == ("x", "y", None)
    assert selection.issues("words") == ()


def test_state_for_project_is_plain_json(selection) -> None:
    import json

    from curvestudio.workflow import ImportIssue

    stored = {
        "columns": ["x", "y", "w"],
        "issues": [{"code": "D", "severity": "warning", "message": "m", "rows": [1]}],
    }
    selection.set_project(_project_with_desktop_state(NUMERIC_FRAME, stored))
    selection.set_columns("words", "x", "w", None)  # a later user choice overrides
    selection.set_issues("words", [ImportIssue(code="E", severity="error", message="bad")])
    state = selection.state_for_project()
    rendered = json.dumps(state, allow_nan=False)  # must be storable in plot_settings
    assert json.loads(rendered) == state
    datasets = state["datasets"]
    assert datasets["words"]["columns"] == ["x", "w", None]
    assert datasets["words"]["issues"][0]["code"] == "E"
