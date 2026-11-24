"""The fit audit through the API: the column a cross-validation is stored in, what a stored
fit's audit and methods say, and the job that cross-validates a fit."""

from pathlib import Path

from alembic import command
from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session

from conftest import make_settings
from curvestudio_api.app import create_app
from curvestudio_api.migrations import _config, current_revision, upgrade_to_head
from curvestudio_api.orm import Dataset, Fit, User


def fit_columns(engine: Engine) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns("fits")}


def test_migration_0012_round_trip(tmp_path: Path) -> None:
    """``fits.cv`` arrives nullable: a fit stored before the column has ``None``, which says
    that no cross-validation has been run. The downgrade takes the column away and nothing else."""
    engine: Engine = create_app(make_settings(tmp_path)).state.ctx.engine
    upgrade_to_head(engine)
    assert current_revision(engine) == "0012"
    assert "cv" in fit_columns(engine)

    command.downgrade(_config(engine), "0011")
    assert current_revision(engine) == "0011"
    assert "cv" not in fit_columns(engine)
    with Session(engine) as db:  # a fit of a database that has not got the column yet
        user = User(email="before@example.org", password_hash="not-a-hash")
        db.add(user)
        db.flush()
        dataset = Dataset(
            user_id=user.id,
            name="line",
            n_rows=0,
            columns=[],
            x_column="x",
            y_column="y",
            fingerprint="f" * 64,
            payload=b"",
        )
        db.add(dataset)
        db.flush()
        fit = Fit(user_id=user.id, dataset_id=dataset.id, spec={}, bundle={}, status="converged")
        db.add(fit)
        db.commit()
        fit_id = fit.id

    upgrade_to_head(engine)
    assert current_revision(engine) == "0012"
    column = next(c for c in inspect(engine).get_columns("fits") if c["name"] == "cv")
    assert column["nullable"] is True
    with Session(engine) as db:  # cv is a deferred column: read it while the session is open
        kept = db.get(Fit, fit_id)
        assert kept is not None and kept.status == "converged"
        assert kept.cv is None
