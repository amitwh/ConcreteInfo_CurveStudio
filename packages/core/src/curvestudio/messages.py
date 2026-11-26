"""One plain sentence per engine status or error code (design item 16).

A status or error code must never reach a user as a bare token: each value
:class:`~curvestudio.fitting.result.FitStatus` can take, each ``error_code`` the API's
jobs fail with, and each code the core's coded error classes carry, has exactly one
plain sentence here, written so the user can act on it. Both apps show the sentence
first and keep the server's or engine's own message after it; neither replaces the
other.

The web cannot import Python, so ``scripts/export_engine_messages.py`` writes this
mapping to ``apps/web/src/content/engine-messages.json`` and a test fails while the two
disagree — this module stays the one authored source. ``tests/unit/test_messages.py``
enumerates the codes from the sources themselves, so a code added anywhere without a
sentence fails the suite.

This module imports nothing, not even the ``FitStatus`` its first four keys mirror:
the desktop shows its window before the numerical stack loads (P6), and this mapping
must not drag ``curvestudio.fitting`` — with scipy, pandas and sympy behind it — into
that window's construction. The test pins the literals to the enum instead.
"""

from collections.abc import Mapping

_SENTENCES: dict[str, str] = {
    # -- fit statuses (the literals of FitStatus; the test fails if the two drift) -------------
    "converged": "The fit converged — its estimates and their uncertainties can be used.",
    "not_converged": (
        "The fit did not converge — try fewer parameters, other starting values or more restarts."
    ),
    "failed": (
        "The fit failed — the engine could not finish it; check the data and the fit "
        "settings and run it again."
    ),
    "invalid_input": (
        "The fit could not run — the data or the settings are not valid; check both and try again."
    ),
    # -- the API's job and envelope codes -----------------------------------------------------
    "TIMEOUT": (
        "The job did not finish in time — simplify it or try again when the server is less busy."
    ),
    "WORKER_CRASHED": (
        "The engine process died while the job ran — run the job again, and try a "
        "smaller or simpler one if it repeats."
    ),
    "RESULT_TOO_LARGE": (
        "The result was too large to store — fit fewer rows or use smaller options "
        "and run it again."
    ),
    "SERVER_RESTART": "The server restarted before the job finished — run the job again.",
    "QUOTA_EXCEEDED": (
        "A daily limit is used up — check the plan's limits and try again tomorrow or "
        "upgrade the plan."
    ),
    "RATE_LIMITED": "Too many requests are being sent at once — wait a moment and try again.",
    "INTERNAL_ERROR": (
        "Something went wrong on the server that should not have — run the job again "
        "and report the problem if it repeats."
    ),
    "FIT_DELETED": (
        "The fit this job needed was deleted — reopen the work from a fit that still exists."
    ),
    "NO_RANKING": (
        "This fit was not opened from an Auto-fit result — open a row of an Auto-fit ranking first."
    ),
    "DATASET_DELETED": "The dataset this job needed was deleted — upload the data again and rerun.",
    # -- the data layer ------------------------------------------------------------------------
    "EMPTY_DATA": "The input holds no data — check the file or the paste and try again.",
    "FILE_NOT_FOUND": "The file was not found — check the path and try again.",
    "FILE_TOO_LARGE": (
        "The file is larger than the server accepts — split it or trim it and upload again."
    ),
    "FILE_UNREADABLE": (
        "The file could not be read — check that it is a valid file and not held open "
        "by another program."
    ),
    "UNREADABLE": "The file could not be read as a table — check its format and try again.",
    "UNSUPPORTED_FORMAT": (
        "Legacy xls workbooks are not supported — save the file as xlsx and import it again."
    ),
    "PARSE_ERROR": "The text could not be parsed — check the encoding and format and try again.",
    "UNKNOWN_SHEET": (
        "The workbook has no sheet by that name — check the sheet name and import again."
    ),
    "TOO_MANY_COLS": (
        "The input has more columns than the server accepts — select fewer columns "
        "and import again."
    ),
    "TOO_MANY_ROWS": (
        "The input has more rows than the server accepts — split it and import the parts."
    ),
    "ROW_OUT_OF_RANGE": "Rows were addressed outside the table — reload the dataset and try again.",
    "DUPLICATE_COLUMN": (
        "A column with that name already exists — rename the new column and import again."
    ),
    "MISSING_VALUE": "A column holds missing values — clean or drop them and try again.",
    "NON_NUMERIC": (
        "A column holds dates or durations rather than numbers — convert it to "
        "numbers and try again."
    ),
    "NON_FINITE": "A column holds infinite values — clean or drop them and try again.",
    "LENGTH_MISMATCH": (
        "The x, y and weight columns differ in length — check the data and try again."
    ),
    "INVALID_X": (
        "The x values must be a non-empty one-dimensional set — check the x column and try again."
    ),
    "INVALID_WEIGHT": (
        "Weights must be zero or more — set a weight to zero to leave an observation "
        "out of the fit."
    ),
    "DOMAIN_ERROR": (
        "The transform is undefined for some values in the column — check those "
        "values or pick another transform."
    ),
    "UNKNOWN_FUNCTION": (
        "An unknown function was used in the transform or equation — check its name "
        "against the supported functions."
    ),
    "UNKNOWN_COLUMN": (
        "A column was named that this analysis does not read — check the column name and try again."
    ),
    "INVALID_SHAPE": ("An input must be one-dimensional — check the data's shape and try again."),
    # -- the workflow layer ---------------------------------------------------------------------
    "INVALID_SPEC": (
        "The fit specification is not valid — check the model, its parameters and the "
        "equation, and try again."
    ),
    "INVALID_EQUATION": (
        "The equation is not valid — check its syntax, parameter names and functions, "
        "and try again."
    ),
    "UNKNOWN_MODEL": (
        "The model id is not in the catalogue — pick a model from the catalogue or "
        "check the custom equation."
    ),
    "INVALID_CUSTOM_MODEL": (
        "A custom equation could not be prepared for fitting — fix the equation and try again."
    ),
    "DUPLICATE_MODEL_ID": (
        "A custom equation uses an id another model already has — give it a unique id "
        "and run Auto-fit again."
    ),
    "DATA_MISMATCH": (
        "The data given are not the data the fit was computed from — reload the "
        "dataset and try again."
    ),
    "MODEL_MISMATCH": (
        "The specification is not the one the fit was computed with — reload the "
        "fit's own specification and try again."
    ),
    "DATA_REQUIRED": (
        "The stored fit keeps only a subsample of its data — provide the full dataset "
        "so the analysis can run."
    ),
    "FIT_NOT_CONVERGED": (
        "This analysis needs a fit that converged — refit until it converges, then "
        "run the analysis."
    ),
    "NOTHING_TO_UNDO": "There is nothing left to undo.",
    "TOO_LARGE": (
        "The request or its data is larger than the server accepts — send a smaller "
        "payload and try again."
    ),
    "TABLE_NOT_STORED": (
        "This dataset was uploaded before whole tables were kept — upload it again to "
        "analyse its columns."
    ),
    # -- Auto-fit --------------------------------------------------------------------------------
    "TOO_FEW_POINTS": (
        "Auto-fit needs more active points than the data holds — check that weights "
        "are positive or collect more data."
    ),
    "UNKNOWN_FAMILY": (
        "One of the selected model families does not exist — uncheck it in the Auto-fit options."
    ),
    "NO_CANDIDATES": (
        "No model fits these settings — widen the selected families or raise the parameter limit."
    ),
    # -- the statistics routines -------------------------------------------------------------------
    "COLLINEAR": (
        "Some columns of the data reproduce each other exactly — drop or combine the "
        "duplicated information and run the analysis again."
    ),
    "CONSTANT_COLUMN": (
        "A group column holds one repeated value, so its spread cannot be used — "
        "remove that column or give it more than one value."
    ),
    "CONSTANT_RESPONSE": (
        "Every y value is identical, so any model with an offset fits it exactly — "
        "check the y column and rerun."
    ),
    "COVARIANCE_NOT_PSD": (
        "The covariance matrix is not usable as it stands, so the result is refused "
        "rather than reported as trustworthy — refit with more data or simpler "
        "settings."
    ),
    "FIT_NOT_CONVERGED_INPUT": (
        "The fit did not converge, so it has no model to cross-validate — refit it first."
    ),
    "GROUP_OF_ONE": (
        "A group holds a single value, so its spread cannot be estimated — group "
        "differently or drop that group."
    ),
    "SINGLE_GROUP": (
        "The group column holds a single group, and this analysis needs at least two "
        "— group differently or pick another analysis."
    ),
    "TOO_FEW_ROWS": (
        "A group holds no usable value, and each group needs the analysis's minimum — "
        "regroup or clean the data."
    ),
    "LEVELS_REQUIRED": (
        "This test needs one of the levels named as its control — pick a control "
        "level and run it again."
    ),
    "INVALID_ARGUMENT": (
        "An argument was passed in a shape the routine cannot take — check the "
        "input's shape and try again."
    ),
    "INVALID_COVARIANCE": (
        "The covariance holds a value that is not a finite number — refit or simplify the model."
    ),
    "INVALID_FOLDS": (
        "The number of folds is not usable — pick a whole number between 2 and the "
        "number of active points."
    ),
    "INVALID_KIND": (
        "The bootstrap kind must be residual or case — pick one of the two and run it again."
    ),
    "INVALID_LEVEL": (
        "The level must lie strictly between 0 and 1 — pick a valid level and try again."
    ),
    "INVALID_MIN_SUCCESS": (
        "The minimum success fraction must lie strictly between 0 and 1 — pick a "
        "valid fraction and try again."
    ),
    "INVALID_N": ("The number of draws must be positive — set a positive number and run it again."),
    "INVALID_N_BOOT": (
        "The number of bootstrap replicates must be at least 50 — raise it and run "
        "the analysis again."
    ),
    "INVALID_OPTION": (
        "An option was passed a value it cannot take — check the value and try again."
    ),
    "INVALID_SEED": (
        "The seed must be a whole number of zero or more — fix the seed and try again."
    ),
    "INTEGRATION_NOT_CONVERGED": (
        "A numerical integral did not reach its tolerance, so the value it stopped at "
        "is not reported as exact — try narrower bounds or simpler inputs."
    ),
    "NO_CONVERGED_REPLICATES": (
        "Too few bootstrap replicates converged to estimate a standard error — raise "
        "the number of replicates or simplify the model."
    ),
    "NO_COVARIANCE": (
        "A parameter has no positive variance, so its profile cannot be searched — "
        "simplify the model or collect more data."
    ),
    "NO_DEGREES_OF_FREEDOM": (
        "The fit leaves no residual direction beyond its parameters — fit fewer "
        "parameters or add data."
    ),
    "NO_STATISTICS": (
        "The fit converged but carries no statistics — refit it so its record is complete."
    ),
    "NO_STATS": (
        "The result carries no fit statistics, so no likelihood is defined — refit and try again."
    ),
    "NO_X_VARIATION": (
        "Every x value is the same, so no curve can be fitted — check the x column and try again."
    ),
    "PERFECT_FIT": (
        "The fit is exact, so likelihood-based criteria are undefined — it is "
        "reported without them."
    ),
    "PROFILE_NOT_SUPPORTED": (
        "A profile likelihood cannot be computed for this fit — use another interval type."
    ),
    "PROPAGATION_UNSTABLE": (
        "More than half the propagation draws failed to give a finite result — "
        "simplify the model or rescale the inputs."
    ),
    "NON_FINITE_RMSE": (
        "The held-out error summed past what floating point can hold — fit fewer "
        "rows or rescale the y values."
    ),
    "RANK_DEFICIENT_DESIGN": (
        "The fit's derivative columns cannot be separated at these points — try "
        "different starting values or fewer parameters."
    ),
    # -- the tools --------------------------------------------------------------------------------
    "NO_DATA": "There are no observations to analyse — load data with at least a few rows.",
    "UNKNOWN_TOOL": "There is no tool by that name — pick one of the listed tools.",
}

#: The one mapping: every sentenced code, in no particular order.
SENTENCES: Mapping[str, str] = _SENTENCES

_FALLBACK = "something went wrong"


def sentence_for(code: str) -> str | None:
    """The one plain sentence for ``code``, or ``None`` when it has none."""
    sentence = SENTENCES.get(code)
    return sentence if sentence is not None and sentence != "" else None


def explain(code: str | None, message: str | None = None) -> str:
    """A code and its own message as the user reads them: the sentence first.

    The sentence leads; a message that adds detail follows it after an em dash, so a
    good server message is never discarded and an unknown code never shows as a bare
    token — the message stands alone when the code has no sentence.
    """
    sentence = sentence_for(code) if code is not None else None
    detail = message.strip() if isinstance(message, str) else ""
    if sentence is None:
        return detail if detail != "" else _FALLBACK
    if detail == "":
        return sentence
    return f"{sentence} — {detail}"


def explain_error(error: BaseException) -> str:
    """A code-carrying engine error as the user reads it (see :func:`explain`).

    Reads the message from each error shape the core defines: ``message`` when the
    class keeps one, the second constructor argument for the two-argument statistics
    errors, and the first for the rest — never the ``str()`` that would embed the code
    the sentence replaces.
    """
    code = getattr(error, "code", None)
    message = getattr(error, "message", None)
    if not isinstance(message, str) or message == "":
        args = error.args
        if len(args) >= 2 and args[0] == code and isinstance(args[1], str):
            message = args[1]
        elif len(args) == 1 and isinstance(args[0], str):
            message = args[0]
        else:
            message = None
    return explain(code if isinstance(code, str) else None, message)
