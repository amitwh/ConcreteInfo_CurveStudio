"""ProjectController: project state, dirty flag, undo, recents — no widgets involved."""

import re

import pytest
from desktop_data import GOOD_CSV, MANY_CSV, TEXT_COLUMN_CSV

from curvestudio.project import Project, ProjectError, load_project
from curvestudio_desktop.controllers.project import ist_now
from curvestudio_desktop.controllers.providers import UnsavedAnswer


def test_new_controller_starts_clean(controller) -> None:
    assert controller.project.datasets == ()
    assert controller.project.name
    assert controller.path is None
    assert controller.is_dirty is False
    assert controller.can_undo is False
    assert controller.recent_files() == []


def test_created_at_is_ist(controller) -> None:
    assert re.fullmatch(r".*\+05:30", controller.project.created_at)
    assert re.fullmatch(r".*\+05:30", ist_now())


def test_import_file_adds_dataset_and_marks_dirty(controller, write_file) -> None:
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path) is True
    assert len(controller.project.datasets) == 1
    entry = controller.project.datasets[0]
    assert entry.name == "points"
    assert controller.is_dirty is True
    assert controller.can_undo is True


def test_import_twice_uniquifies_ids(controller, write_file) -> None:
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    assert controller.import_dataset_file(path)
    ids = [entry.id for entry in controller.project.datasets]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_unreadable_file_becomes_messenger_call(controller, write_file, messenger) -> None:
    path = write_file("data.csv", b"\xff\xfe\x00 not utf8 \x9c")
    assert controller.import_dataset_file(path) is False
    assert len(messenger.errors) == 1
    title, message = messenger.errors[0]
    assert "Cannot import" in title
    assert "UNREADABLE" in message
    assert controller.project.datasets == ()
    assert controller.is_dirty is False


def test_unsupported_extension_is_messaged(controller, write_file, messenger) -> None:
    path = write_file("table.xyz", "1,2\n3,4\n")
    assert controller.import_dataset_file(path) is False
    assert any("UNSUPPORTED_FORMAT" in message for _, message in messenger.errors)


def test_import_with_error_issues_is_refused(controller, write_file, messenger) -> None:
    """The dialog blocks error data, but the controller must stay safe if it gets some."""
    path = write_file("bad.csv", TEXT_COLUMN_CSV)
    assert controller.import_dataset_file(path) is False
    assert any("NON_NUMERIC" in message for _, message in messenger.errors)
    assert controller.project.datasets == ()
    assert controller.is_dirty is False


def test_import_bytes_rejects_oversized_paste(controller, messenger) -> None:
    huge = b"x,y\n" + b"1,2\n" * 5_250_000  # ~21 MB, beyond the default byte limit
    assert controller.import_dataset_bytes(huge, "clipboard.csv") is False
    assert any("TOO_LARGE" in message for _, message in messenger.errors)


def test_import_bytes_accepts_small_paste(controller) -> None:
    assert controller.import_dataset_bytes(b"a,b\n1,2.0\n2,4.1\n3,6.2\n", "clipboard.csv")
    assert len(controller.project.datasets) == 1


def test_remove_dataset(controller, write_file) -> None:
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    entry_id = controller.project.datasets[0].id
    assert controller.remove_dataset(entry_id) is True
    assert controller.project.datasets == ()
    assert controller.remove_dataset("missing") is False


def test_rename_dataset(controller, write_file, messenger) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    entry_id = controller.project.datasets[0].id
    assert controller.rename_dataset(entry_id, "  Beam A  ") is True
    assert controller.project.datasets[0].name == "Beam A"
    assert controller.rename_dataset(entry_id, "   ") is False
    assert any("name" in message.lower() for _, message in messenger.errors)


def test_undo_restores_previous_state(controller, write_file) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    entry_id = controller.project.datasets[0].id
    controller.rename_dataset(entry_id, "renamed")
    assert controller.undo() is True
    assert controller.project.datasets[0].name == "points"
    assert controller.undo() is True
    assert controller.project.datasets == ()
    assert controller.undo() is False


def test_open_save_roundtrip(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    target = tmp_path / "demo.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save_as() is True
    assert controller.path == target
    assert controller.is_dirty is False
    assert target.is_file()

    controller.rename_dataset(controller.project.datasets[0].id, "changed")
    assert controller.is_dirty is True
    controller.confirmation.answers.append(UnsavedAnswer.DISCARD)
    assert controller.open_project(target) is True
    assert controller.project.datasets[0].name == "points"
    assert controller.is_dirty is False
    assert controller.path == target


def test_failed_open_keeps_unsaved_work(controller, write_file, tmp_path, messenger) -> None:
    """A corrupt file must never destroy the current (dirty) project."""
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    broken = write_file("broken.cstudio", b"definitely not a zip archive")
    assert controller.open_project(broken) is False
    assert any("open" in title.lower() for title, _ in messenger.errors)
    assert "Traceback" not in messenger.error_text()
    assert len(controller.project.datasets) == 1
    assert controller.is_dirty is True


def test_open_cancel_keeps_current_project(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    target = tmp_path / "other.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save_as() is True  # a real .cstudio to open
    controller.import_dataset_file(write_file("more.csv", GOOD_CSV))  # make it dirty again
    controller.paths.open_paths.append(target)
    controller.confirmation.answers.append(UnsavedAnswer.CANCEL)
    assert controller.open_project() is False
    assert len(controller.project.datasets) == 2
    assert controller.is_dirty is True
    assert controller.confirmation.actions  # the prompt carried a human-readable action


def test_open_missing_file_is_messaged(controller, tmp_path, messenger) -> None:
    assert controller.open_project(tmp_path / "nope.cstudio") is False
    assert len(messenger.errors) == 1
    assert controller.project.datasets == ()


def test_save_without_path_asks_for_one(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    target = tmp_path / "chosen.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True
    assert controller.path == target
    assert controller.is_dirty is False


def test_save_as_cancelled_changes_nothing(controller) -> None:
    assert controller.save_as() is False  # no scripted path: user cancelled
    assert controller.path is None


def test_save_failure_is_messaged_and_dirty_stays(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    controller.paths.save_paths.append(tmp_path / "missing-dir" / "out.cstudio")
    assert controller.save_as() is False
    assert any("save" in title.lower() for title, _ in controller.messenger.errors)
    assert controller.is_dirty is True


def test_new_project_confirmation_paths(controller, write_file) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    controller.confirmation.answers.append(UnsavedAnswer.CANCEL)
    assert controller.new_project() is False
    assert len(controller.project.datasets) == 1

    controller.confirmation.answers.append(UnsavedAnswer.DISCARD)
    assert controller.new_project() is True
    assert controller.project.datasets == ()
    assert controller.is_dirty is False
    assert controller.path is None
    assert controller.can_undo is False


def test_new_project_save_answer_saves_first(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    target = tmp_path / "keep.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.path is None
    controller.confirmation.answers.append(UnsavedAnswer.SAVE)
    assert controller.new_project() is True
    assert target.is_file()
    assert load_project(target).datasets[0].name == "points"


def test_new_project_save_cancelled_aborts(controller, write_file) -> None:
    path = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(path)
    controller.confirmation.answers.append(UnsavedAnswer.SAVE)  # save() then asks for a path
    assert controller.new_project() is False  # path dialog cancelled -> abort everything
    assert len(controller.project.datasets) == 1
    assert controller.is_dirty is True


def test_recent_files_ordered_deduped_and_capped(controller, write_file, tmp_path) -> None:
    paths = []
    for index in range(12):
        csv = write_file(f"d{index}.csv", GOOD_CSV)
        assert controller.import_dataset_file(csv)
        target = tmp_path / f"p{index}.cstudio"
        controller.paths.save_paths.append(target)
        assert controller.save_as() is True
        paths.append(target)
    recent = controller.recent_files()
    assert recent == list(reversed(paths))[:10]  # newest first, at most ten

    controller.paths.save_paths.append(paths[3])  # re-saving promotes without duplicating
    assert controller.save_as() is True
    assert controller.recent_files()[0] == paths[3]
    assert len(controller.recent_files()) == 10


def test_recent_files_prune_missing(controller, write_file, tmp_path) -> None:
    csv = write_file("points.csv", GOOD_CSV)
    controller.import_dataset_file(csv)
    first = tmp_path / "first.cstudio"
    second = tmp_path / "second.cstudio"
    controller.paths.save_paths.append(first)
    controller.save_as()
    controller.paths.save_paths.append(second)
    controller.save_as()
    first.unlink()
    assert controller.recent_files() == [second]


def test_signals_fire_on_state_changes(controller, write_file, qtbot, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    with qtbot.waitSignal(controller.datasets_changed):
        assert controller.import_dataset_file(path)
    dirty_seen: list[bool] = []
    controller.dirty_changed.connect(dirty_seen.append)
    target = tmp_path / "out.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True
    assert dirty_seen == [False]


def test_project_type_is_the_core_project(controller) -> None:
    assert isinstance(controller.project, Project)


def test_load_only_and_adopt_decompose_open(controller, write_file, tmp_path) -> None:
    """The background-open path: load without touching state, then adopt."""
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    target = tmp_path / "async.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save_as() is True

    loaded = controller.load_only(target)
    assert loaded.datasets[0].name == "points"
    assert controller.project is not loaded  # nothing adopted before confirmation

    assert controller.adopt_opened(loaded, target) is True
    assert controller.path == target
    assert controller.is_dirty is False
    assert controller.project.datasets[0].name == "points"


def test_load_only_raises_for_a_broken_archive(controller, write_file) -> None:
    broken = write_file("broken.cstudio", b"not a zip")
    with pytest.raises(ProjectError):
        controller.load_only(broken)


def test_adopt_opened_respects_a_cancel(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    target = tmp_path / "keep.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save_as() is True
    controller.import_dataset_file(write_file("more.csv", GOOD_CSV))  # dirty again

    loaded = controller.load_only(target)
    controller.confirmation.answers.append(UnsavedAnswer.CANCEL)
    assert controller.adopt_opened(loaded, target) is False
    assert len(controller.project.datasets) == 2  # the current work survives
    assert controller.is_dirty is True


def test_write_only_and_commit_save_decompose_saving(controller, write_file, tmp_path) -> None:
    """The background-save path: write first, commit state only on success."""
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    target = tmp_path / "async.cstudio"

    written = controller.write_only(target)
    assert target.is_file()
    assert controller.is_dirty is True  # not committed yet
    assert written.datasets[0].name == "points"

    controller.commit_save(written, target)
    assert controller.is_dirty is False
    assert controller.path == target


def test_write_only_raises_project_error(controller, tmp_path) -> None:
    with pytest.raises(ProjectError):
        controller.write_only(tmp_path / "missing-dir" / "out.cstudio")


def test_saved_project_carries_desktop_state_via_provider(controller, write_file, tmp_path) -> None:
    path = write_file("points.csv", GOOD_CSV)
    assert controller.import_dataset_file(path)
    entry_id = controller.project.datasets[0].id
    controller.set_state_provider(
        lambda: {"datasets": {entry_id: {"columns": ["x", "y", None], "issues": []}}}
    )
    target = tmp_path / "stated.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save_as() is True
    blob = controller.project.plot_settings["desktop"]
    assert blob["datasets"][entry_id]["columns"] == ["x", "y", None]


@pytest.mark.parametrize(
    "answer", [UnsavedAnswer.DISCARD, UnsavedAnswer.SAVE, UnsavedAnswer.CANCEL]
)
def test_every_answer_is_honoured(controller, write_file, tmp_path, answer) -> None:
    controller.import_dataset_file(write_file("first.csv", GOOD_CSV))
    target = tmp_path / "p.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save_as() is True  # current path becomes target
    controller.import_dataset_file(write_file("second.csv", MANY_CSV))  # dirty again
    controller.paths.open_paths.append(target)
    controller.confirmation.answers.append(answer)
    result = controller.open_project()
    assert result is (answer is not UnsavedAnswer.CANCEL)
    if answer is UnsavedAnswer.SAVE:
        assert len(load_project(target).datasets) == 2  # the second import was saved first
    elif answer is UnsavedAnswer.DISCARD:
        assert len(controller.project.datasets) == 1  # back to the saved version
        assert controller.is_dirty is False
    else:  # CANCEL leaves everything untouched
        assert len(controller.project.datasets) == 2
        assert controller.is_dirty is True
