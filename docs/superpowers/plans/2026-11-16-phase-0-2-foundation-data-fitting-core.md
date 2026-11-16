# CurveStudio Phases 0–2 (Foundation, Data, Fitting Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the repository foundation, the data layer, and a fitting engine that is validated against NIST certified reference data and usable from a CLI.

**Architecture:** A `uv` workspace whose first package, `curvestudio-core`, is pure Python (NumPy/SciPy/LMfit/SymPy/Pandas/Pydantic) and never imports a GUI or web framework. Models are declarative `ModelSpec`s; every fit returns one immutable, JSON-serializable `FitResult` with full provenance. Correctness is proven by NIST StRD benchmarks, not by eyeballing.

**Tech Stack:** Python 3.14, uv, NumPy, SciPy, LMfit, SymPy, Pandas, openpyxl, Pydantic 2, pytest, Hypothesis, Ruff, MyPy, import-linter, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-16-curvestudio-design.md` (design) and `docs/intent.md` (PRD). Read both before starting.

## Global Constraints

- Python **3.14** floor; `uv` environments; dependencies pinned in `uv.lock`. Before adding any dependency, check its latest stable version and OSV/GitHub advisories and put source + date in that commit message (versions seen 16/11/2026: numpy 2.5.5, scipy 1.18.1, lmfit 1.3.4, pandas 3.1.1, pydantic 2.14.3, openpyxl 3.1.5, pytest 9.2.0, ruff 0.17.5, mypy 2.4.1, import-linter 2.16, hypothesis 6.170.15 — re-check at install).
- `curvestudio-core` imports only NumPy, SciPy, LMfit, Statsmodels, SymPy, Pandas, Pydantic (+ openpyxl for XLSX). **Never** PySide6, FastAPI or React. Statsmodels is not needed until Phase 3; do not add it now.
- Licence: **AGPL-3.0-or-later** for every package; one `LICENSE` at the repo root; `license = "AGPL-3.0-or-later"` in every `pyproject.toml`.
- Never `eval`/`exec`/`pickle` user-supplied content. Never silently delete observations: every cleaning step is a recorded, reversible operation. A fit that fails is reported as failed — never replaced by a plausible-looking result.
- Timestamps: timezone-aware ISO 8601 with `+05:30` offset (IST).
- Weights `w_i` multiply squared residuals (minimise Σ w_i·r_i²).
- Information-criteria convention is deferred to Phase 3; do not compute AIC/BIC here.
- Commit message ends with the line `Amit Haridas`. **After every commit run `git push`.** Each task = tests green first, then commit and push.
- Tests live in root `tests/{unit,integration,numerical}/`; numerical benchmarks that take >5 s carry `@pytest.mark.slow`.

## Review Focus

1. Data containing NaN/blank cells → fit **refuses** with `MISSING_VALUE` naming the rows; it never silently drops them (Task 1.1 `Dataset.xy`, Task 1.2).
2. Fewer observations than parameters (`n <= k`) → `status="invalid_input"`, no standard errors, no crash (Task 2.4).
3. Noiseless data (zero residuals): R²=1, SSE=0, no divide-by-zero warnings escaping, covariance defined or `None` (Task 2.2, 2.3).
4. Constant X, or X on wildly different scales (1e-9 and 1e9): clear error / ill-conditioning warning, correct polynomial result when conditioning allows (Task 1.2, 2.3).
5. CSV with UTF-8 BOM, trailing blank rows/columns, duplicate or blank header names, thousands separators: imports predictably and records what it changed (Task 1.3).

---

## Phase 0 — Foundation

### Task 0.1: Workspace scaffold and licence

**Files:**
- Create: `pyproject.toml` (workspace root + ruff/mypy/pytest config), `packages/core/pyproject.toml`, `packages/core/src/curvestudio/__init__.py`, `LICENSE` (AGPL-3.0 text), `README.md`, `.python-version`, `tests/unit/test_package.py`
- Modify: `.gitignore` (add `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `*.egg-info/`, `dist/`)

**Interfaces:**
- Produces: importable package `curvestudio` with `curvestudio.__version__: str == "0.1.0.dev0"`; `uv run pytest` works from the root; ruff line length 100, mypy `strict = true` for `curvestudio`.

- [ ] **Step 1: Write failing test** `tests/unit/test_package.py::test_version_is_exposed` asserting `curvestudio.__version__ == "0.1.0.dev0"`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_package.py -v` → FAIL (module missing).
- [ ] **Step 3: Create the workspace** — root `pyproject.toml` with `[tool.uv.workspace] members=["packages/*"]`, dev dependency group (pytest, pytest-cov, hypothesis, ruff, mypy, import-linter), and pytest markers `slow`; `packages/core` with the dependencies listed in Global Constraints (no Statsmodels), `requires-python = ">=3.14"`. README states what the project is, the AGPL licence, and that it is pre-alpha.
- [ ] **Step 4: Run** `uv sync && uv run pytest -v && uv run ruff check . && uv run mypy packages/core/src` → all pass; `uv.lock` is created.
- [ ] **Step 5: Commit** (`feat: scaffold uv workspace and curvestudio-core package`) and push.

### Task 0.2: Architecture guard, CI and community files

**Files:**
- Create: `.importlinter` (or `[tool.importlinter]` in root pyproject), `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.github/ISSUE_TEMPLATE/{bug_report,model_request}.md`, `.github/pull_request_template.md`, `tests/unit/test_architecture.py`

**Interfaces:**
- Consumes: package `curvestudio` from Task 0.1.
- Produces: `uv run lint-imports` fails if anything under `curvestudio` imports `PySide6`, `fastapi`, `sqlalchemy`, `starlette` or `uvicorn`; CI job `ci` on `ubuntu-latest` and `windows-latest` running sync (frozen), ruff check, ruff format --check, mypy, lint-imports, `pytest -m "not slow"`.

- [ ] **Step 1: Write failing test** `test_core_has_no_gui_or_web_imports`: scans every `.py` under `packages/core/src` with `ast` and asserts none imports a forbidden top-level module; add a second test that feeds the checker a synthetic snippet `import PySide6` and asserts it is flagged (so the guard itself is tested).
- [ ] **Step 2: Run** → FAIL (checker not written). **Step 3:** implement the checker in the test module plus the import-linter forbidden-modules contract. **Step 4:** run the test and `uv run lint-imports` → pass.
- [ ] **Step 5: Write** CONTRIBUTING (dev setup, `uv`, inbound=outbound AGPL contribution statement, rule that every model needs equation, parameter definitions, references, tests and assumptions — PRD §24), CODE_OF_CONDUCT (Contributor Covenant 2.1 text), SECURITY (private reporting via GitHub advisories), templates, and `ci.yml` (Python 3.14).
- [ ] **Step 6: Commit** (`ci: add architecture guard, CI matrix and community files`), push, then `gh run watch` the workflow; the commit is not done until CI is green on both OSes. Fix forward if not.

---

## Phase 1 — Data

### Task 1.1: Dataset and operations log

**Files:**
- Create: `packages/core/src/curvestudio/data/{__init__,dataset,operations,errors}.py`
- Test: `tests/unit/data/test_dataset.py`

**Interfaces:**
- Produces (all in `curvestudio.data`):
  - `class DataError(Exception)` with `.code: str` and `.rows: tuple[int, ...]`.
  - Operations (frozen dataclasses): `ExcludeRows(rows: tuple[int, ...])`, `TransformColumn(source: str, target: str, func: Literal["log","log10","sqrt","reciprocal","square","affine"], scale: float = 1.0, offset: float = 0.0)`, `RenameColumn(old: str, new: str)`. `Operation = ExcludeRows | TransformColumn | RenameColumn`.
  - `@dataclass(frozen=True) class XY: x: np.ndarray; y: np.ndarray; weights: np.ndarray | None; x_name: str; y_name: str; fingerprint: str` (arrays are float64, 1-D, read-only).
  - `@dataclass(frozen=True) class Dataset: name: str; original: pd.DataFrame; ops: tuple[Operation, ...] = (); import_notes: tuple[str, ...] = ()` with `table() -> pd.DataFrame` (original with ops applied; excluded rows kept), `active_mask() -> np.ndarray` (bool, False for excluded), `with_op(op) -> Dataset`, `undo() -> Dataset` (drops the last op; error if none), `fingerprint() -> str` (SHA-256 hex over column names, dtypes and values of the **original** plus the op list), `xy(x: str, y: str, weights: str | None = None) -> XY`.

- [ ] **Step 1: Write failing tests:** (a) `with_op(ExcludeRows((1,)))` leaves `original` byte-identical and `active_mask()[1] is False`; (b) `undo()` restores the previous `fingerprint()`; (c) `TransformColumn("x","lx","log")` on `[1, e]` gives `[0, 1]` to 1e-12; (d) `xy("x","y")` on data with a NaN in an active row raises `DataError` with `code == "MISSING_VALUE"` and `rows == (that row,)`, while excluding that row first makes it succeed (**Review Focus 1**); (e) `xy` rejects `inf` with code `NON_FINITE`; (f) the same data constructed twice has equal fingerprints, and a one-ULP change in one value changes it; (g) mutating the returned `XY` arrays raises `ValueError` (read-only).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** per Interfaces; `original` is stored as a copy and never mutated; ops apply in order against a copy. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** (`feat(data): immutable Dataset with reversible operations log`) and push.

### Task 1.2: Validation

**Files:**
- Create: `packages/core/src/curvestudio/data/validation.py`
- Test: `tests/unit/data/test_validation.py`

**Interfaces:**
- Consumes: `Dataset` (Task 1.1).
- Produces: `@dataclass(frozen=True) class Issue: code: str; severity: Literal["error","warning"]; message: str; rows: tuple[int, ...] = ()`; `validate(ds: Dataset, x: str, y: str, weights: str | None = None, positive_required: tuple[str, ...] = ()) -> list[Issue]`. Codes: `NON_NUMERIC`, `MISSING_VALUE`, `NON_FINITE`, `DUPLICATE_ROW`, `INSUFFICIENT_DATA` (<3 active rows, error), `CONSTANT_X` (error), `NONPOSITIVE_FOR_LOG` (error, only for columns named in `positive_required`), `INVALID_WEIGHT` (any weight ≤ 0 or non-finite, error). Only active (non-excluded) rows are checked. `NON_NUMERIC` comes from `Dataset.import_notes`/coerced cells recorded by the importer (Task 1.3) — here, test it with a hand-built `Dataset` whose notes contain `"coerced:x:3"`.

- [ ] **Step 1: Write failing tests**, one per code, each asserting `code`, `severity` and exact `rows`; plus `test_constant_x_is_error` and `test_validate_clean_dataset_returns_empty_list`; `test_excluded_rows_not_reported`; `test_x_range_1e-9_to_1e9_warns` expecting a warning `WIDE_DYNAMIC_RANGE` when max|x|/min|x≠0| > 1e12 (**Review Focus 4**).
- [ ] **Step 2–4:** run (FAIL), implement `validate`, run (PASS).
- [ ] **Step 5: Commit** (`feat(data): validation issues for fitting readiness`) and push.

### Task 1.3: Importers

**Files:**
- Create: `packages/core/src/curvestudio/data/importers.py`, `tests/data/import/{basic.csv,bom_blank.csv,semicolon.csv,thousands.csv,sample.xlsx}` (generate `sample.xlsx` in a fixture with openpyxl rather than committing a binary)
- Test: `tests/unit/data/test_importers.py`

**Interfaces:**
- Consumes: `Dataset`, `DataError`.
- Produces: `@dataclass(frozen=True) class ImportLimits: max_bytes: int = 50_000_000; max_rows: int = 1_000_000; max_cols: int = 1_000`; `read_dataset(source: str | os.PathLike[str], *, fmt: str | None = None, sheet: str | int = 0, limits: ImportLimits = ImportLimits()) -> Dataset` (format from extension: csv, tsv, txt, json, xlsx; `.xls` raises `DataError("UNSUPPORTED_FORMAT")` with a message to convert to xlsx); `read_text(text: str, *, name: str = "pasted") -> Dataset` (clipboard/pasted tables; sniffs delimiter among `,` `\t` `;` space).
- Behaviour decided here: strip a UTF-8 BOM; drop fully blank rows/columns and say so in `import_notes`; blank/duplicate header names become `col_N` / `name_2` and are noted; thousands separators (`1,234.5`) are parsed only in columns whose every non-blank cell matches that pattern, else treated as text; non-numeric cells in an otherwise numeric column become NaN **and** add a note `coerced:<column>:<row>` (they are never dropped); XLSX opened with `openpyxl` `read_only=True, data_only=True`; limits are enforced **before** parsing (`DataError("FILE_TOO_LARGE")` / `TOO_MANY_ROWS`).

- [ ] **Step 1: Write failing tests:** CSV/TSV/TXT/JSON/XLSX each load the same 5-row table equal to a literal expected frame; BOM + blank trailing rows/cols file loads with correct shape and notes listing what was dropped (**Review Focus 5**); a cell `"abc"` in a numeric column yields NaN plus note `coerced:y:2`; a file over `max_bytes` (set limit to 10) raises `FILE_TOO_LARGE`; `read_text` handles a pasted tab-separated block; an `.xls` path raises `UNSUPPORTED_FORMAT`.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS).
- [ ] **Step 5: Commit** (`feat(data): CSV/TSV/TXT/JSON/XLSX/clipboard importers with safety limits`) and push.

---

## Phase 2 — Fitting core

### Task 2.1: ModelSpec, registry and contract checker

**Files:**
- Create: `packages/core/src/curvestudio/models/{__init__,spec,registry}.py`, `packages/core/src/curvestudio/testing/{__init__,model_contract}.py`
- Test: `tests/unit/models/test_spec.py`, `tests/unit/models/test_registry.py`

**Interfaces:**
- Produces (`curvestudio.models`):
  - `@dataclass(frozen=True) class Parameter: name: str; description: str = ""; unit: str = ""; lower: float = -inf; upper: float = inf`.
  - `@dataclass(frozen=True) class ModelSpec: id: str; name: str; category: str; expression: str; parameters: tuple[Parameter, ...]; initial_guess: Callable[[np.ndarray, np.ndarray], dict[str, float]]; variables: tuple[str, ...] = ("x",); domain_check: Callable[[np.ndarray, np.ndarray], list[str]] | None = None; references: tuple[str, ...] = (); assumptions: str = ""` with methods `evaluate(x: np.ndarray, **params: float) -> np.ndarray`, `jacobian(x: np.ndarray, **params: float) -> np.ndarray` (shape `(n, k)`, columns in `parameters` order), `latex() -> str`, `equation_text(params: Mapping[str, float], digits: int = 6) -> str` (the equation with fitted numbers substituted, e.g. `y = 2.38942*(1 - exp(-0.000550156*x))`), `applicability(x, y) -> list[str]` (reasons the model does not apply; empty = applicable). `expression` is the right-hand side in SymPy syntax; it is parsed **once at construction** with `sympy.parse_expr` using a whitelisted namespace (`exp, log, ln→log, log10, sqrt, sin, cos, tan, sinh, cosh, Abs, pi, E`) and `evaluate=False`-safe transformations only; unknown names raise `ValueError`. Evaluator and Jacobian come from `sympy.lambdify(..., "numpy")` on the expression and its symbolic derivatives.
  - Registry: `register(spec: ModelSpec) -> None` (duplicate id raises `ValueError`), `get_model(model_id: str) -> ModelSpec` (`KeyError` with close-match hint), `list_models(category: str | None = None) -> list[ModelSpec]`.
  - `curvestudio.testing.model_contract.check_model_contract(spec: ModelSpec, x: np.ndarray, true_params: dict[str, float], *, seed: int = 0) -> None` asserting: analytic Jacobian equals central finite differences (rtol 1e-6), `evaluate` returns finite float64 of the right shape, `initial_guess` returns every parameter within bounds, and `latex()` is non-empty. Phase 4 reuses this for every model.

- [ ] **Step 1: Write failing tests:** spec `a*exp(b*x)` evaluates to `2*exp(0.5*[0,1])` at those points; its Jacobian equals the hand-derived `[exp(b x), a x exp(b x)]`; `equation_text({"a":2.0,"b":0.5})` returns `"y = 2*exp(0.5*x)"`; a spec with expression `__import__('os').system('x')` and one with `a*foo(x)` both raise `ValueError` at construction; duplicate `register` raises; `get_model("expo")` error suggests the near match; `check_model_contract` passes for the exponential and fails (raises `AssertionError`) for a spec whose hand-supplied Jacobian is wrong (use a subclass/patch in the test).
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). **Step 5: Commit** (`feat(models): declarative ModelSpec, registry and contract checker`) and push.

### Task 2.2: Metrics, FitResult and serialization

**Files:**
- Create: `packages/core/src/curvestudio/fitting/{__init__,metrics,result}.py`
- Test: `tests/unit/fitting/test_metrics.py`, `tests/unit/fitting/test_result.py`

**Interfaces:**
- Produces (`curvestudio.fitting`):
  - `fit_statistics(y: np.ndarray, fitted: np.ndarray, n_params: int, weights: np.ndarray | None = None, through_origin: bool = False) -> FitStatistics` where `FitStatistics` (Pydantic, frozen) has `n, k, df, sse, mse, rmse, mae, r2, adj_r2, residual_se, r2_definition: Literal["centered","uncentered"]` (`mse = sse/n`, `rmse = sqrt(mse)`, `residual_se = sqrt(sse/df)` or `None` when `df <= 0`, `adj_r2 = 1-(1-r2)*(n-1)/df` (centred) or `None` when `df <= 0`). `r2` is centred (`1 - sse/sst`); with `through_origin=True` it is the uncentred form and `r2_definition` is `"uncentered"`. Weighted: `sse = Σ w r²`, `sst` weighted about the weighted mean. When `sst == 0` and `sse == 0`, `r2 = 1.0`; when `sst == 0` and `sse > 0`, `r2 = None`.
  - `class FitStatus(StrEnum): CONVERGED="converged"; NOT_CONVERGED="not_converged"; FAILED="failed"; INVALID_INPUT="invalid_input"`.
  - `class ParameterEstimate(BaseModel)`: `name, value, std_err | None, ci_low | None, ci_high | None`.
  - `class Provenance(BaseModel)`: `software_versions: dict[str,str]` (curvestudio, numpy, scipy, lmfit, python), `dataset_fingerprint: str`, `created_at: str` (IST ISO 8601, `+05:30`), `seed: int`.
  - `class FitResult(BaseModel, frozen)`: `model_id, equation, equation_fitted: str | None, parameters: tuple[ParameterEstimate, ...], covariance: list[list[float]] | None, correlation: list[list[float]] | None, stats: FitStatistics | None, fitted: list[float], residuals: list[float], status: FitStatus, termination_reason: str, method: str, loss: str, initial_params: dict[str,float], n_starts: int, warnings: tuple[str, ...], provenance: Provenance` with `to_json() -> str` and `@classmethod from_json(s: str) -> FitResult` (lossless round-trip, `NaN`/`inf` never emitted — use `None`).

- [ ] **Step 1: Write failing tests:** known case `y=[1,2,3,4]`, `fitted=[1.1,1.9,3.2,3.8]`, `k=2` gives `sse=0.1`, `rmse=sqrt(0.025)`, `mae=0.125`, `r2=1-0.1/5=0.98`, `adj_r2=1-0.02*3/2=0.97` (assert each to 1e-12); noiseless `fitted==y` gives `r2==1.0`, `sse==0.0` and emits no `RuntimeWarning` (run under `warnings.simplefilter("error")`) (**Review Focus 3**); `n == k` gives `df=0`, `residual_se is None`, `adj_r2 is None`; weighted known case against a hand calculation; `FitResult.from_json(r.to_json()) == r`; `created_at` ends with `+05:30`.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). **Step 5: Commit** (`feat(fitting): fit statistics and serializable FitResult`) and push.

### Task 2.3: Linear and polynomial regression (OLS/WLS)

**Files:**
- Create: `packages/core/src/curvestudio/fitting/linear.py`
- Test: `tests/unit/fitting/test_linear.py`, `tests/numerical/test_nist_linear.py` (stub is added in Task 2.8; here use inline Wampler1/Norris data)

**Interfaces:**
- Consumes: `XY`, `FitResult`, `FitStatistics`, `ParameterEstimate`, `Provenance`.
- Produces: `fit_polynomial(data: XY, degree: int, *, through_origin: bool = False, confidence_level: float = 0.95) -> FitResult` (`degree=1` is simple linear regression; weights from `data.weights`; `model_id` `"poly{degree}"` or `"poly{degree}_origin"`). Coefficients named `b0..b{degree}`. Standard errors from `s²(XᵀWX)⁻¹`; confidence intervals use the Student-t quantile with `df`. Solve with QR/`lstsq` on a column-scaled design matrix (never the normal equations). Warnings (strings in `FitResult.warnings`): degree ≥ 6 → `"HIGH_ORDER_POLYNOMIAL: unstable outside the data range"`; condition number of the scaled design > 1e12 → `"ILL_CONDITIONED"`. `n <= k` → `status=INVALID_INPUT`, `termination_reason` explains, no SEs. Exact-fit data gives `covariance` zeros, not NaN.

- [ ] **Step 1: Write failing tests:** Norris (`degree=1`) matches certified `B0=-0.262323073774029`, `B1=1.00211681802045`, SE(B1) `4.29796848199937e-4`, residual SD `0.884796396144373`, R² `0.999993745883712` to 9 significant digits (embed the 36 rows as a fixture read from `tests/data/nist_strd/linear/Norris.dat` once Task 2.7 exists — until then inline them; copy the rows from the NIST file, do not retype); Wampler1 (`degree=5`, y = 1+x+x²+x³+x⁴+x⁵ at x=0..20) recovers all six coefficients as 1 to 1e-8 with `r2 == 1.0` (**Review Focus 3**); weighted fit with weights `[1,1,1,100]` on `y=[1,2,3,10]` equals a hand-computed WLS solution (compute it in the test with `np.linalg.lstsq` on `sqrt(w)`-scaled rows); `through_origin=True` on `y=2x` gives slope 2 with `r2_definition == "uncentered"`; 3 points degree 3 → `INVALID_INPUT`; x spanning 1e-9..1e9 at degree 3 returns a result whose warnings contain `ILL_CONDITIONED` or whose coefficients are still correct to 1e-6 (**Review Focus 4**); degree 7 includes the high-order warning.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). **Step 5: Commit** (`feat(fitting): OLS/WLS linear and polynomial regression`) and push.

### Task 2.4: Nonlinear fitting engine

**Files:**
- Create: `packages/core/src/curvestudio/fitting/{config,methods,nonlinear,api}.py`
- Test: `tests/unit/fitting/test_nonlinear.py`, `tests/unit/fitting/test_methods.py`

**Interfaces:**
- Consumes: `ModelSpec`, `XY`, `FitResult` and friends.
- Produces:
  - `@dataclass(frozen=True) class FitConfig: method: str = "auto"; loss: str = "linear"; initial: dict[str, float] | None = None; bounds: dict[str, tuple[float, float]] | None = None; n_starts: int = 1; seed: int = 0; max_nfev: int | None = None; ftol: float = 1e-10; xtol: float = 1e-10; gtol: float = 1e-10; confidence_level: float = 0.95`.
  - `check_method_compatibility(method: str, loss: str, has_bounds: bool, has_jacobian: bool) -> None` raising `ValueError` with a clear message for invalid combinations (e.g. `"lm"` with bounds or with loss ≠ `"linear"`; unknown method). `"auto"` resolves to `"lm"` without bounds and `"trf"` with bounds, and the resolved name is what `FitResult.method` records.
  - `fit(model: ModelSpec | str, data: XY, config: FitConfig = FitConfig()) -> FitResult` — the single public fitting entry point. Linear-in-parameters polynomial specs are *not* special-cased here; callers use `fit_polynomial`.
- Behaviour decided here: start from `config.initial` else `model.initial_guess(x, y)`; clip starts into bounds; with `n_starts > 1`, extra starts are drawn from a `numpy.random.default_rng(seed)` as log-uniform/uniform perturbations of the base start inside bounds, and the lowest-SSE converged run wins (all starts' SSE recorded in `warnings` if they disagree beyond 1e-6 relative); covariance = `s²(JᵀWJ)⁻¹` using the model's analytic Jacobian (pseudo-inverse and a `"SINGULAR_COVARIANCE"` warning when rank-deficient; SEs `None` then); `status` is `CONVERGED` only if the optimizer reports success **and** all parameters finite **and** the Jacobian at the solution is finite; otherwise `NOT_CONVERGED` with the optimizer's message in `termination_reason`; exceptions inside the optimizer give `FAILED` (message kept, never swallowed); `n <= k` or `model.applicability(x, y)` non-empty → `INVALID_INPUT` (**Review Focus 2**); `data.weights` honoured.

- [ ] **Step 1: Write failing tests:** (a) the PRD §20 example — `x = linspace(0,10,100)`, `y = 5*exp(-0.4x)+2 + N(0,0.1)` with `default_rng(42)` — fit `a*exp(-b*x)+c` and assert `a≈5, b≈0.4, c≈2` within 5 % and `status == CONVERGED`; (b) same fit twice with the same seed gives identical `to_json()` apart from `created_at`; (c) `method="lm"` with `bounds` raises `ValueError` mentioning "bounds"; (d) `n_starts=5` from a deliberately bad start (`b=-5`) converges where `n_starts=1` returns `NOT_CONVERGED` (choose the bad start so this is deterministic; record it in the test); (e) 2 points for a 3-parameter model → `INVALID_INPUT`, `parameters == ()`, `stats is None`, no exception; (f) data that cannot be fit (`y` constant against `a*exp(b*x)` with `max_nfev=3`) → `NOT_CONVERGED`, never `CONVERGED`; (g) perfectly noiseless data yields `sse < 1e-20`, `r2 == 1.0`, no warnings raised as exceptions (**Review Focus 3**); (h) a model whose `applicability` says `x > 0` required, given x containing 0 → `INVALID_INPUT` naming the reason.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). Use `scipy.optimize.least_squares` through `lmfit.Minimizer` so parameters, bounds and `fixed` handling stay in LMfit. **Step 5: Commit** (`feat(fitting): nonlinear least-squares engine with multi-start`) and push.

### Task 2.5: Robust, global and derivative-free methods

**Files:**
- Modify: `packages/core/src/curvestudio/fitting/{methods,nonlinear}.py`
- Test: `tests/unit/fitting/test_methods_extra.py`

**Interfaces:**
- Consumes: `FitConfig`, `check_method_compatibility`, `fit` (Task 2.4).
- Produces: `fit` accepts `method` in `{"auto","lm","trf","dogbox","nelder","powell","bfgs","lbfgsb","differential_evolution","dual_annealing"}` and `loss` in `{"linear","soft_l1","huber","cauchy"}`. Rules: robust losses only with `trf`/`dogbox`/`"auto"` (auto resolves to `trf` when `loss != "linear"`); `differential_evolution` and `dual_annealing` **require finite bounds on every parameter** (else `ValueError`); `nelder`, `powell`, `bfgs`, `lbfgsb` minimise the weighted SSE; `bfgs`/`lbfgsb` use the analytic Jacobian-based gradient; `nelder` and `powell` accept bounds (supported by SciPy), `bfgs` rejects them with a `ValueError`. For all non-least-squares methods the covariance/SEs are computed from the analytic Jacobian at the solution. Seeds are passed to the stochastic methods and recorded.

- [ ] **Step 1: Write failing tests:** with 3 gross outliers added to linear data (`y = 2x+1`, outliers at +50), `loss="soft_l1"` recovers slope within 2 % while `loss="linear"` is off by more than 10 % (proves robustness is real); `differential_evolution` with bounds recovers `a*exp(-b*x)+c` from the §2.4 data within 5 %; with `seed` fixed two runs are identical; DE without bounds raises `ValueError`; `bfgs`, `lbfgsb`, `nelder`, `powell`, `dogbox`, `dual_annealing` each converge on the §2.4 data (parametrised, loose 10 % tolerance); `"lm"` + `huber` raises `ValueError`.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). **Step 5: Commit** (`feat(fitting): robust, global and derivative-free fit methods`) and push.

### Task 2.6: Starter model set

**Files:**
- Create: `packages/core/src/curvestudio/models/starter.py`
- Modify: `packages/core/src/curvestudio/models/__init__.py` (register on import)
- Test: `tests/unit/models/test_starter.py`

**Interfaces:**
- Consumes: `ModelSpec`, `register`, `check_model_contract`.
- Produces: registered specs with these ids and `expression`s: `exp_growth` `a*exp(b*x)`; `exp_decay_offset` `a*exp(-b*x)+c`; `exp_offset` `a+b*exp(c*x)`; `log_natural` `a+b*log(x)`; `log_base10` `a+b*log10(x)`; `power` `a*x**b`; `michaelis_menten` `vmax*x/(km+x)`; `logistic3` `k/(1+exp(-r*(x-x0)))`; `gaussian` `a*exp(-(x-mu)**2/(2*s**2))`. Each has bounds (e.g. `b`>0 where required), a data-derived `initial_guess` (linearised or heuristic, deterministic), `domain_check` (e.g. `x>0` for log/power), a one-line `assumptions`, and at least one reference string.

- [ ] **Step 1: Write failing test** parametrised over every starter id: generate noiseless data from true params, call `check_model_contract`, then `fit` from `initial_guess` and assert recovered params within 1e-4 relative and `status == CONVERGED`. Add `test_log_models_reject_nonpositive_x`.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS). **Step 5: Commit** (`feat(models): starter models with data-derived initial guesses`) and push.

### Task 2.7: NIST StRD reference data and parser

**Files:**
- Create: `scripts/fetch_nist_strd.py`, `tests/data/nist_strd/{nonlinear,linear}/*.dat`, `tests/data/nist_strd/README.md`, `packages/core/src/curvestudio/testing/nist.py`
- Test: `tests/unit/testing/test_nist_parser.py`

**Interfaces:**
- Produces: `fetch_nist_strd.py` downloads (once; files are then committed) from `https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/<Name>.dat` and `https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/<Name>.dat`. Nonlinear: `Misra1a, Misra1b, Misra1c, Misra1d, DanWood, Chwirut2, Lanczos3, Gauss1, BoxBOD, Rat42, MGH09, Eckerle4`. Linear: `Norris, Pontius, Wampler1, Wampler4, Filip`. `README.md` records source URL, retrieval date, and that NIST data is US-government public domain.
- `curvestudio.testing.nist`: `@dataclass(frozen=True) class NistProblem: name: str; difficulty: Literal["lower","average","higher"] | None; x: np.ndarray; y: np.ndarray; start1: dict[str,float]; start2: dict[str,float]; certified: dict[str,float]; certified_se: dict[str,float]; certified_sse: float; certified_residual_sd: float; df: int; model_formula: str`; `load_nls(path: Path) -> NistProblem`; `@dataclass(frozen=True) class NistLinearProblem` (`name, x, y, certified, certified_se, residual_sd, r2, degree`); `load_lls(path: Path) -> NistLinearProblem` (degree inferred from the parameter count: `B0..Bk`); `lre(estimate: float, certified: float) -> float` = `-log10(|estimate-certified|/|certified|)` capped at 15 (and `15` when equal; absolute error if certified is 0).

- [ ] **Step 1: Write failing tests** against the committed files: `load_nls(Misra1a.dat)` has 14 observations, `difficulty == "lower"`, `start1 == {"b1":500,"b2":1e-4}`, `start2 == {"b1":250,"b2":5e-4}`, `certified["b1"] == 2.3894212918e+02`, `certified_se["b2"] == 7.2668688436e-06`, `certified_sse == 1.2455138894e-01`, `df == 12`; `load_lls(Norris.dat)` gives 36 rows and `certified["B1"] == 1.00211681802045`; `lre(1.0, 1.0)==15`, `lre(1.001, 1.0)≈3`.
- [ ] **Step 2: Run** → FAIL. **Step 3:** write the fetch script (urllib, timeout, fails loudly on non-200), run it **once** with network access and commit the downloaded files unchanged; implement the parser against the actual file layout (`Starting values`/`Certified Values` block, `Data:` header row, scientific notation incl. `E0` suffixes). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** (`test: vendor NIST StRD datasets and parser`) and push.

### Task 2.8: NIST benchmark suite and validation report

**Files:**
- Create: `tests/numerical/test_nist_nonlinear.py`, `tests/numerical/test_nist_linear.py`, `tests/numerical/nist_models.py`, `scripts/nist_report.py`, `docs/numerical-validation.md`
- Modify: `tests/unit/fitting/test_linear.py` (switch Norris/Wampler1 to the vendored files)

**Interfaces:**
- Consumes: `load_nls`, `load_lls`, `lre`, `fit`, `fit_polynomial`, `ModelSpec`, `XY`.
- Produces: `nist_models.py` maps each nonlinear problem name to a `ModelSpec` built from its formula (parameters named `b1..bk` exactly as NIST), e.g. `Misra1a: b1*(1-exp(-b2*x))`, `DanWood: b1*x**b2`, `Chwirut2: exp(-b1*x)/(b2+b3*x)`, `BoxBOD: b1*(1-exp(-b2*x))`, `Rat42: b1/(1+exp(b2-b3*x))`, `MGH09: b1*(x**2+x*b2)/(x**2+x*b3+b4)`, `Eckerle4: (b1/b2)*exp(-0.5*((x-b3)/b2)**2)`, etc. — **verify each expression against the `Model:` line inside its `.dat` file; the file is authoritative.** `scripts/nist_report.py` runs every problem × both starts and writes a Markdown table (problem, difficulty, start, status, min LRE over parameters, LRE of SE, SSE LRE) to `docs/numerical-validation.md` with the date and library versions.
- **Acceptance rules (the tests):** (1) **No silently wrong answers:** whenever `status == CONVERGED`, min LRE over parameters must be ≥ 5 and SSE LRE ≥ 7 — a converged result that misses is a test failure. (2) Difficulty `lower`: both starts must converge with min LRE ≥ 6. (3) Difficulty `average`/`higher`: each start either converges (then rule 1 applies) or is reported `NOT_CONVERGED`; additionally run each with `n_starts=20, seed=0` and bounds of ±1000×|start value| per parameter, and record the converged fraction. Only rule 1 is asserted for these; the pass-rate is written to the report, not hidden. (4) Linear: Norris, Pontius, Wampler1 converge to LRE ≥ 9 on coefficients; Wampler4 and Filip must either reach LRE ≥ 6 or carry an `ILL_CONDITIONED` warning — never silent garbage.

- [ ] **Step 1: Write the failing parametrised tests** implementing rules (1)–(4) over all vendored problems (mark the full sweep `@pytest.mark.slow`; keep a fast subset — Misra1a, DanWood, Norris, Wampler1 — unmarked so CI exercises it).
- [ ] **Step 2: Run** `uv run pytest tests/numerical -v` → expect FAIL initially for models not yet mapped; **Step 3:** write `nist_models.py`; fix any engine defects the benchmark reveals **in the engine, not by loosening the tolerances** (if a rule genuinely cannot be met for a named hard problem, mark that single case `xfail(strict=True)` with the reason and list it in the report).
- [ ] **Step 4:** run the full sweep, run `scripts/nist_report.py`, commit the generated `docs/numerical-validation.md`. **Step 5: Commit** (`test(numerical): NIST StRD benchmark suite and validation report`) and push.

### Task 2.9: CLI vertical slice

**Files:**
- Create: `packages/core/src/curvestudio/cli.py`, `examples/decay.csv` (the PRD §20 example: `x = linspace(0,10,100)`, `y = 5*exp(-0.4x)+2 + N(0,0.1)`, `default_rng(42)`, columns `x,y`), `examples/README.md`
- Modify: `packages/core/pyproject.toml` (`[project.scripts] curvestudio = "curvestudio.cli:main"`)
- Test: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: `read_dataset`, `validate`, `Dataset.xy`, `fit`, `fit_polynomial`, `get_model`, `list_models`.
- Produces: `main(argv: list[str] | None = None) -> int` with subcommands `fit` (`curvestudio fit FILE --x COL --y COL --model ID | --poly N [--weights COL] [--method M] [--loss L] [--starts N] [--seed S] [--json OUT]`) and `models` (lists ids, names, categories). Exit codes: `0` converged, `2` invalid input / validation errors (printed, one per line with row numbers), `3` not converged or failed. Output: equation with fitted numbers, a parameter table (value, SE, 95 % CI), R², adjusted R², RMSE, status line, warnings; `--json` writes `FitResult.to_json()`.

- [ ] **Step 1: Write failing tests:** end-to-end on `examples/decay.csv` with `--model exp_decay_offset` returns exit 0 and the output contains `R²` and recovered `a` within 5 % of 5; a CSV with a blank y cell exits 2 and the output names the row (**Review Focus 1**); `--json` file round-trips through `FitResult.from_json`; `--model nope` exits 2 with a near-match hint; an unconverged fit exits 3.
- [ ] **Step 2–4:** run (FAIL), implement with `argparse`, run (PASS). **Step 5: Commit** (`feat(cli): fit command vertical slice (import → validate → fit → report)`) and push.

### Phase gate

- [ ] Run `uv run ruff check . && uv run ruff format --check . && uv run mypy packages/core/src && uv run lint-imports && uv run pytest` (include `-m slow` once) → all green; grep the diff for `TODO|FIXME|stub|placeholder|for now` → none.
- [ ] Confirm CI green on Linux and Windows (`gh run list -L 3`).
- [ ] Update `README.md` status section with the numerical-validation summary line; commit and push (`docs: Phase 0–2 status`).
- [ ] Report to the owner: what shipped, the NIST pass-rate, any `xfail` problems, and ask approval to write the Phase 3 (statistics) plan.
