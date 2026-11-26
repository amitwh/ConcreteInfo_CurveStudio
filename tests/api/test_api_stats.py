"""The statistical tools through the API: the catalogue, running a tool, stored results."""

import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
from alembic import command
from api_test_helpers import linear_csv, make_settings, signup, upload
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, inspect, select

from curvestudio.data import read_dataset
from curvestudio.tools.stats import FAMILIES, build_request, catalogue, run_tool
from curvestudio_api.app import create_app
from curvestudio_api.migrations import _config, current_revision, upgrade_to_head
from curvestudio_api.orm import AuditLog, Dataset, StatResult

API = "/api/v1"

STAT_CHOICES: dict[str, dict[str, object]] = {
    "anova": {"values": "a", "groups": "g"},
    "kruskal": {"values": "a", "groups": "g"},
    "twoway": {"values": "out", "factor_a": "fa", "factor_b": "fb"},
    "repeated": {"measures": ["m1", "m2", "m3"]},
    "friedman": {"measures": ["m1", "m2", "m3"]},
    "blandaltman": {"method_a": "a", "method_b": "b"},
    "deming": {"method_a": "a", "method_b": "b"},
    "passingbablok": {"method_a": "a", "method_b": "b"},
    "contingency": {"rows": "rowg", "columns": "colg"},
    "mcnemar": {"before": "before", "after": "after"},
    "correlation": {"columns": ["a", "b", "out"]},
    "descriptives": {"values": "a", "groups": "g"},
    "distribution": {"values": "a"},
    "ttest": {"kind": "one_sample", "a": "a"},
    "power": {"design": "one_sample", "effect_size": 0.5, "power": 0.8},
    "mannwhitney": {"a": "a", "b": "b"},
    "wilcoxon": {"a": "a", "b": "b"},
    "signtest": {"a": "a", "b": "b"},
    "regression": {"outcome": "out", "predictors": ["p1", "p2"]},
}
"""One run of every tool: numeric columns for values, text and date-like columns for groups."""


def stats_csv(n: int = 60) -> bytes:
    """A table with numeric, text and date-like columns that every tool can run on."""
    rng = np.random.default_rng(11)
    lines = ["x,y,w,a,b,m1,m2,m3,out,p1,p2,g,fa,fb,rowg,colg,before,after,d"]
    for i in range(n):
        x = float(i + 1)
        numbers = (
            x,
            0.5 * x + rng.normal(scale=0.05),
            1.0,
            1.0 + 0.1 * i + rng.normal(scale=0.02),
            1.0 + 0.1 * i + rng.normal(scale=0.03),
            1.0 + 0.05 * i,
            1.1 + 0.06 * i,
            1.2 + 0.04 * i,
            2.0 + rng.normal(),
            rng.normal(),
            rng.normal(),
        )
        cells = (
            "ABC"[i % 3],
            "AB"[i % 2],
            "XY"[(i // 2) % 2],
            "yes" if i % 2 == 0 else "no",
            ["yes", "yes", "no", "no"][i % 4],
            "yes" if i % 4 in (0, 1) else "no",
            "yes" if i % 4 in (0, 1, 2) else "no",
            "2024-01-01" if i % 2 == 0 else "2024-01-02",
        )
        lines.append(",".join([repr(value) for value in numbers] + list(cells)))
    return ("\n".join(lines) + "\n").encode()


def _upload_stats_table(client: TestClient, headers: dict[str, str]) -> dict[str, object]:
    created = upload(
        client,
        headers,
        stats_csv(),
        "stats.csv",
        x_column="x",
        y_column="y",
        weight_column="w",
    )
    assert created.status_code == 201, created.text
    body: dict[str, object] = created.json()
    return body


def _audit_actions(client: TestClient, result_id: str) -> list[str]:
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        return [
            row.action
            for row in db.scalars(
                select(AuditLog)
                .where(AuditLog.entity_id == result_id)
                .order_by(AuditLog.created_at, AuditLog.id)
            )
        ]


def test_the_catalogue_is_the_core_s(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = shared_client.get(f"{API}/stats/tools", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["families"] == [[key, title] for key, title in FAMILIES]
    assert body["tools"] == [info.model_dump(mode="json") for info in catalogue()]


def test_every_tool_runs_and_its_numbers_are_the_core_s(
    shared_client: TestClient, tmp_path: Path
) -> None:
    headers = signup(shared_client)
    dataset = _upload_stats_table(shared_client, headers)
    path = tmp_path / "stats.csv"
    path.write_bytes(stats_csv())
    core = read_dataset(path)

    for tool, choices in STAT_CHOICES.items():
        response = shared_client.post(
            f"{API}/datasets/{dataset['id']}/stats/{tool}", headers=headers, json=choices
        )
        assert response.status_code == 201, (tool, response.text)
        body = response.json()
        expected = run_tool(
            tool, build_request(tool, choices, column=core.column, group=core.group)
        )
        dump = expected.model_dump(mode="json")
        assert body["result"] == dump, tool
        assert body["request"] == choices
        assert body["title"] == expected.title
        with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
            row = db.get(StatResult, uuid.UUID(body["id"]))
            assert row is not None, tool
            assert row.result == dump, tool
            assert row.tool == tool
            assert str(row.dataset_id) == dataset["id"]


def test_a_result_is_stored_listed_read_and_deleted(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv(30)).json()
    url = f"{API}/datasets/{dataset['id']}/stats/descriptives"
    first = shared_client.post(url, headers=headers, json={"values": "y"})
    assert first.status_code == 201, first.text
    time.sleep(0.002)
    second = shared_client.post(url, headers=headers, json={"values": "y"})
    assert second.status_code == 201, second.text

    listing = shared_client.get(
        f"{API}/stats/results", headers=headers, params={"dataset_id": dataset["id"]}
    )
    assert listing.status_code == 200
    summaries = listing.json()
    assert [item["id"] for item in summaries] == [second.json()["id"], first.json()["id"]]
    assert "result" not in summaries[0]
    assert summaries[0]["title"] == second.json()["title"]

    read = shared_client.get(f"{API}/stats/results/{first.json()['id']}", headers=headers)
    assert read.status_code == 200
    assert read.json() == first.json()
    assert _audit_actions(shared_client, first.json()["id"]) == ["stat.create"]

    removed = shared_client.delete(f"{API}/stats/results/{first.json()['id']}", headers=headers)
    assert removed.status_code == 204
    assert (
        shared_client.get(f"{API}/stats/results/{first.json()['id']}", headers=headers).status_code
        == 404
    )
    assert _audit_actions(shared_client, first.json()["id"]) == ["stat.create", "stat.delete"]
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.get(StatResult, uuid.UUID(first.json()["id"])) is None


def test_another_users_result_is_not_found(shared_client: TestClient) -> None:
    a = signup(shared_client)
    b = signup(shared_client)
    dataset = upload(shared_client, a, linear_csv()).json()
    created = shared_client.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives", headers=a, json={"values": "y"}
    )
    assert created.status_code == 201, created.text
    result_id = created.json()["id"]
    for method in ("GET", "DELETE"):
        response = shared_client.request(method, f"{API}/stats/results/{result_id}", headers=b)
        assert response.status_code == 404, (method, response.text)
        assert response.json()["error"] == "NOT_FOUND"
    assert (
        shared_client.get(
            f"{API}/stats/results", headers=b, params={"dataset_id": dataset["id"]}
        ).status_code
        == 404
    )
    assert shared_client.get(f"{API}/stats/results", headers=b).json() == []


def _insert_old_dataset(client: TestClient, headers: dict[str, str]) -> str:
    """A dataset stored before its whole table was kept: fit payload only, no table."""
    me = client.get(f"{API}/me", headers=headers).json()
    dataset_id = uuid.uuid4()
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        db.add(
            Dataset(
                id=dataset_id,
                user_id=uuid.UUID(me["id"]),
                name="old",
                n_rows=3,
                columns=["x", "y", "g"],
                x_column="x",
                y_column="y",
                weight_column=None,
                issues=[],
                fingerprint="f" * 64,
                payload=b"x,y\n1,2\n2,4\n3,6\n",
            )
        )
        db.commit()
    return str(dataset_id)


def test_a_dataset_from_before_offers_only_the_fit_s_columns(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset_id = _insert_old_dataset(shared_client, headers)
    url = f"{API}/datasets/{dataset_id}/stats"

    fit_column = shared_client.post(f"{url}/descriptives", headers=headers, json={"values": "y"})
    assert fit_column.status_code == 201, fit_column.text

    other = shared_client.post(f"{url}/descriptives", headers=headers, json={"values": "g"})
    assert other.status_code == 422, other.text
    assert other.json()["detail"]["code"] == "TABLE_NOT_STORED"
    assert "upload it again" in other.json()["message"]


def test_an_unknown_tool_and_column(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = _upload_stats_table(shared_client, headers)
    url = f"{API}/datasets/{dataset['id']}/stats"

    unknown_tool = shared_client.post(f"{url}/nope", headers=headers, json={"values": "a"})
    assert unknown_tool.status_code == 404
    assert unknown_tool.json()["error"] == "NOT_FOUND"

    unknown_column = shared_client.post(
        f"{url}/descriptives", headers=headers, json={"values": "nope"}
    )
    assert unknown_column.status_code == 422
    assert unknown_column.json()["detail"]["code"] == "UNKNOWN_COLUMN"

    one_group = upload(shared_client, headers, b"x,y,g\n1,2,same\n2,4,same\n3,6,same\n").json()
    refused = shared_client.post(
        f"{API}/datasets/{one_group['id']}/stats/anova",
        headers=headers,
        json={"values": "x", "groups": "g"},
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["code"] == "SINGLE_GROUP"


def test_the_row_cap_is_checked_before_anything_is_read(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory(stats_max_rows=5))  # type: ignore[operator]
    headers = signup(client)
    dataset = upload(client, headers, linear_csv(10)).json()
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        row = db.get(Dataset, uuid.UUID(dataset["id"]))
        assert row is not None
        # Neither payload can be read; a refusal that read one would fail with a 500.
        row.payload = b"not a stored payload"
        row.table = b"not a stored table"
        db.commit()
    response = client.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives",
        headers=headers,
        json={"values": "x"},
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["detail"]["code"] == "TOO_MANY_ROWS"
    assert "5" in body["message"]
    assert "desktop app and the command line" in body["message"]


def test_a_tool_with_its_own_cap(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv(1001)).json()
    response = shared_client.post(
        f"{API}/datasets/{dataset['id']}/stats/passingbablok",
        headers=headers,
        json={"method_a": "x", "method_b": "y"},
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["detail"]["code"] == "TOO_MANY_ROWS"
    assert "1,000" in body["message"]
    assert "1,001" in body["message"]


def test_the_quota_counts_stored_results(
    client_factory: Callable[..., TestClient], settings_factory: Callable[..., object]
) -> None:
    client = client_factory(settings_factory())  # type: ignore[operator]
    headers = signup(client)
    me = client.get(f"{API}/me", headers=headers).json()
    user_id = uuid.UUID(me["id"])
    dataset = upload(client, headers, linear_csv(30)).json()
    before = client.get(f"{API}/me", headers=headers).json()["usage"]["dataset_bytes"]

    created = client.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives", headers=headers, json={"values": "y"}
    )
    assert created.status_code == 201, created.text
    after = client.get(f"{API}/me", headers=headers).json()["usage"]["dataset_bytes"]
    with client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        row = db.get(StatResult, uuid.UUID(created.json()["id"]))
        assert row is not None
        assert row.size_bytes > 0
    assert after == before + row.size_bytes

    tight = client_factory(settings_factory(quota_max_dataset_bytes=after + 1))  # type: ignore
    refused = tight.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives", headers=headers, json={"values": "y"}
    )
    assert refused.status_code == 413, refused.text
    assert refused.json()["error"] == "QUOTA_EXCEEDED"
    with tight.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        stored = select(func.count(StatResult.id)).where(StatResult.user_id == user_id)
        assert db.scalar(stored) == 1
        assert (
            db.scalar(
                select(func.count(AuditLog.id)).where(
                    AuditLog.user_id == user_id, AuditLog.action == "stat.create"
                )
            )
            == 1
        )


def test_a_tool_without_data_needs_no_dataset(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = shared_client.post(
        f"{API}/stats/power",
        headers=headers,
        json={"design": "one_sample", "effect_size": 0.5, "power": 0.8},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["dataset_id"] is None
    assert body["result"]["tool"] == "power"
    listing = shared_client.get(f"{API}/stats/results", headers=headers).json()
    assert body["id"] in [item["id"] for item in listing]


def test_a_tool_with_data_is_refused_without_one(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    response = shared_client.post(
        f"{API}/stats/descriptives", headers=headers, json={"values": "x"}
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["detail"]["code"] == "INVALID_OPTION"
    assert "needs a dataset" in body["message"]


def test_deleting_a_dataset_or_an_account_removes_its_results(shared_client: TestClient) -> None:
    headers = signup(shared_client)
    dataset = upload(shared_client, headers, linear_csv()).json()
    created = shared_client.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives", headers=headers, json={"values": "y"}
    )
    assert created.status_code == 201, created.text
    removed = shared_client.delete(f"{API}/datasets/{dataset['id']}", headers=headers)
    assert removed.status_code == 204
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.get(StatResult, uuid.UUID(created.json()["id"])) is None

    other = signup(shared_client)
    me = shared_client.get(f"{API}/me", headers=other).json()
    dataset = upload(shared_client, other, linear_csv()).json()
    created = shared_client.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives", headers=other, json={"values": "y"}
    )
    assert created.status_code == 201, created.text
    erased = shared_client.request(
        "DELETE", f"{API}/me", headers=other, json={"confirm": me["email"].split("@")[0]}
    )
    assert erased.status_code == 204, erased.text
    with shared_client.app.state.ctx.session_factory() as db:  # type: ignore[attr-defined]
        assert db.get(StatResult, uuid.UUID(created.json()["id"])) is None
        assert (
            db.scalar(
                select(func.count(StatResult.id)).where(StatResult.user_id == uuid.UUID(me["id"]))
            )
            == 0
        )


def test_group_labels_match_the_core_accessor(shared_client: TestClient, tmp_path: Path) -> None:
    content = (
        b"x,y,g\n"
        b"1,2, A \n"
        b"2,4,A\n"
        b"3,6,\n"
        b"4,8,2\n"
        b"5,10,2.0\n"
        b"6,12,B\n"
        b"7,14,B\n"
        b"8,16,2024-01-02\n"
        b"9,18,2024-01-02\n"
        b"10,20,B\n"
    )
    headers = signup(shared_client)
    created = upload(shared_client, headers, content, x_column="x", y_column="y")
    assert created.status_code == 201, created.text
    dataset = created.json()
    response = shared_client.post(
        f"{API}/datasets/{dataset['id']}/stats/descriptives",
        headers=headers,
        json={"values": "y", "groups": "g"},
    )
    assert response.status_code == 201, response.text
    table = next(
        item
        for item in response.json()["result"]["tables"]
        if any(column["key"] == "group" for column in item["columns"])
    )
    labels = [row[0] for row in table["rows"]]

    path = tmp_path / "groups.csv"
    path.write_bytes(content)
    core = read_dataset(path).group("g")
    assert labels == list(dict.fromkeys(core.labels.tolist()))


def test_migration_0015_round_trip(tmp_path: Path) -> None:
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    assert current_revision(engine) == "0016"
    assert "stat_results" in inspect(engine).get_table_names()

    command.downgrade(_config(engine), "0014")
    assert current_revision(engine) == "0014"
    assert "stat_results" not in inspect(engine).get_table_names()

    upgrade_to_head(engine)
    assert current_revision(engine) == "0016"
    assert "stat_results" in inspect(engine).get_table_names()
    columns = {column["name"]: column for column in inspect(engine).get_columns("stat_results")}
    assert set(columns) == {
        "id",
        "user_id",
        "dataset_id",
        "tool",
        "request",
        "result",
        "size_bytes",
        "created_at",
        "updated_at",
    }
    assert columns["user_id"]["nullable"] is False
    assert columns["dataset_id"]["nullable"] is True
