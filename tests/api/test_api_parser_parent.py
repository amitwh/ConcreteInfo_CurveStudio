"""The warm parser parent: uploads fork from a preloaded server (P6 Task 6, half A).

The guarantees Task 5 established are untouched — one fresh, self-capped child per upload —
so those tests stay as they are. What changes is where the child comes from on Linux: a
forkserver this process started with the parsing stack already imported. These tests hold
the new part: starting the parent is idempotent, and a real upload parses through it.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import linear_csv, signup, upload
from curvestudio_api.importer import PARSER_START_METHOD, start_parser_parent
from curvestudio_api.settings import Settings

LINUX = Path("/proc/self/stat").exists()


def test_starting_the_parent_twice_starts_it_once() -> None:
    start_parser_parent()
    if PARSER_START_METHOD != "forkserver":
        pytest.skip("the warm parser parent is the Linux forkserver")
    from multiprocessing import forkserver

    server = forkserver._forkserver._forkserver_pid
    start_parser_parent()
    assert forkserver._forkserver._forkserver_pid == server


def test_an_upload_parses_through_the_warm_parent(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., Settings]
) -> None:
    client = client_factory(settings_factory())
    headers = signup(client)
    response = upload(client, headers, linear_csv(30))
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["n_rows"] == 30
