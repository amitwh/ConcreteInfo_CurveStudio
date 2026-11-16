# CurveStudio

CurveStudio is an open-source curve-fitting and data-analysis application, an alternative to
CurveExpert. It fits models to experimental data, ranks candidate models, and reports fit
statistics, with a reusable engine (`curvestudio-core`) that has no GUI or web dependencies.

**Status: pre-alpha.** Interfaces and results are not stable and must not be relied on yet.

## Licence

CurveStudio is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). See [LICENSE](LICENSE).

## Development

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy packages/core/src
```
