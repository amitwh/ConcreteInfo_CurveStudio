"""The engine's codes reach the user as sentences (design item 16).

Every code the core or the API can hand a UI — each ``FitStatus``, every literal the
API sets as a job's ``error_code``, and every constant code the core's code-carrying
error classes are raised with — has exactly one plain sentence in
``curvestudio.messages``, so a code never reaches a user as a bare token. The codes are
enumerated from the sources themselves (an AST scan), so a new code fails here until it
is sentenced.
"""

import ast
import json
from pathlib import Path

from curvestudio.fitting.result import FitStatus
from curvestudio.messages import SENTENCES, explain, explain_error, sentence_for

REPO = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
    REPO / "packages" / "core" / "src" / "curvestudio",
    REPO / "apps" / "api" / "src" / "curvestudio_api",
)
WEB_MESSAGES = REPO / "apps" / "web" / "src" / "content" / "engine-messages.json"

#: The error classes whose ``code`` a UI can end up showing, by bare name: the tools,
#: the statistics routines, the workflow layer, Auto-fit's own refusals and the data
#: layer all raise these with a constant code as the first argument.
CODED_ERRORS = frozenset(
    {"ToolError", "StatisticsError", "WorkflowError", "AutoFitError", "DataError"}
)


def _constant(node: ast.AST) -> str | None:
    """A plain string constant, or ``None`` for anything dynamic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def codes_the_sources_can_raise() -> set[str]:
    """Every statically known code the core and the API can attach to an error or job.

    Raised codes are the first argument of the coded error classes (a dynamic first
    argument is skipped: whatever it passes was itself raised somewhere covered); the
    API's job codes are ``error_code=`` keyword literals and module constants they name.
    """
    codes: set[str] = set()
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            module_constants: dict[str, str] = {}
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.isupper()
                ):
                    value = _constant(node.value)
                    if value is not None:
                        module_constants[node.targets[0].id] = value
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name in CODED_ERRORS and node.args:
                    code = _constant(node.args[0])
                    if code is not None:
                        codes.add(code)
                if name == "_check_fraction" and len(node.args) >= 2:
                    code = _constant(node.args[1])
                    if code is not None:
                        codes.add(code)
                for keyword in node.keywords:
                    if keyword.arg != "error_code":
                        continue
                    if isinstance(keyword.value, ast.Name):
                        codes.add(module_constants.get(keyword.value.id, ""))
                    else:
                        code = _constant(keyword.value)
                        if code is not None:
                            codes.add(code)
    codes.discard("")
    return codes


#: Envelope codes the UIs surface beside a message (quotas and rate limits are shown as
#: codes today, although they never travel as a job's ``error_code``).
ENVELOPE_CODES = frozenset({"QUOTA_EXCEEDED", "RATE_LIMITED"})


def test_every_code_the_sources_can_raise_has_exactly_one_sentence() -> None:
    required = codes_the_sources_can_raise() | ENVELOPE_CODES
    missing = sorted(required - SENTENCES.keys())
    assert missing == [], f"codes without a sentence: {missing}"


def test_every_fit_status_has_a_sentence() -> None:
    missing = sorted(str(status) for status in FitStatus if str(status) not in SENTENCES)
    assert missing == [], f"FitStatus values without a sentence: {missing}"
    # The module spells the statuses as literals (it imports nothing, so the desktop's
    # window builds before the numerical stack loads): a renamed value must not leave
    # its old literal behind. Status values are lowercase; every code is uppercase.
    lowercase = sorted(key for key in SENTENCES if key.islower())
    assert lowercase == sorted(str(status) for status in FitStatus), (
        f"lowercase keys that are no fit status: {lowercase}"
    )


def test_each_sentence_is_one_plain_sentence() -> None:
    for code, sentence in SENTENCES.items():
        assert sentence.count(".") == 1, f"{code}: more than one sentence: {sentence!r}"
        assert sentence.endswith("."), f"{code}: does not end with a period: {sentence!r}"
        assert "\n" not in sentence, f"{code}: not a single line: {sentence!r}"
        assert sentence[0].isupper(), f"{code}: does not start capitalized: {sentence!r}"


def test_no_two_codes_share_a_sentence() -> None:
    seen: dict[str, str] = {}
    for code, sentence in sorted(SENTENCES.items()):
        previous = seen.setdefault(sentence, code)
        assert previous == code, f"{code} and {previous} share the sentence {sentence!r}"


def test_the_web_copy_of_the_mapping_is_in_sync() -> None:
    """The JSON the web imports must be exactly the core's mapping, regenerated by
    ``scripts/export_engine_messages.py`` — the one authored source is the core."""
    shipped = json.loads(WEB_MESSAGES.read_text(encoding="utf-8"))
    assert shipped == dict(SENTENCES), (
        "apps/web/src/content/engine-messages.json is stale; run "
        "`uv run scripts/export_engine_messages.py` and commit the result"
    )


def test_sentence_for_and_explain() -> None:
    assert sentence_for("NO_SUCH_CODE") is None
    assert sentence_for("TIMEOUT") == SENTENCES["TIMEOUT"]
    # the sentence leads, the server's own message still surfaces after it
    assert explain("TIMEOUT", "the fit did not finish within 30 s") == (
        f"{SENTENCES['TIMEOUT']} — the fit did not finish within 30 s"
    )
    assert explain("TIMEOUT", None) == SENTENCES["TIMEOUT"]
    assert explain(None, "plain message") == "plain message"
    assert explain(None, None) == "something went wrong"
    assert explain("NO_SUCH_CODE", "plain message") == "plain message"


def test_explain_error_reads_each_error_shape() -> None:
    from curvestudio.data.errors import DataError
    from curvestudio.discovery.types import AutoFitError
    from curvestudio.statistics.errors import StatisticsError
    from curvestudio.tools._errors import ToolError
    from curvestudio.workflow.errors import WorkflowError

    tools = explain_error(ToolError("NO_DATA", "the bundle carries no observations"))
    assert tools == f"{SENTENCES['NO_DATA']} — the bundle carries no observations"
    # WorkflowError's str() embeds the code; the sentence must not be doubled by it
    workflow = explain_error(WorkflowError("INVALID_SPEC", "the model has no parameters"))
    assert workflow == f"{SENTENCES['INVALID_SPEC']} — the model has no parameters"
    # StatisticsError keeps its message in args[1], DataError in args[0]
    stats = explain_error(StatisticsError("INVALID_LEVEL", "level must lie in (0, 1)"))
    assert stats == f"{SENTENCES['INVALID_LEVEL']} — level must lie in (0, 1)"
    data = explain_error(DataError("MISSING_VALUE", "column y holds NaN values"))
    assert data == f"{SENTENCES['MISSING_VALUE']} — column y holds NaN values"
    autofit = explain_error(AutoFitError("TOO_FEW_POINTS", "needs at least 4 points"))
    assert autofit == f"{SENTENCES['TOO_FEW_POINTS']} — needs at least 4 points"
    # an error without a code reads as its message alone
    assert explain_error(ValueError("plain failure")) == "plain failure"
