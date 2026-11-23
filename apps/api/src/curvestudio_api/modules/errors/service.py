"""The rules for error events: scrubbing, roll-up by fingerprint, and the caps on what a row
holds. Nothing here knows about HTTP or logging: the router and ``capture`` bring events in."""

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from curvestudio_api.modules.errors.models import ErrorEvent
from curvestudio_api.timeutil import utcnow

MAX_ERROR_TYPE = 200
MAX_MESSAGE = 2_000
MAX_STACK = 100_000
MAX_CONSOLE_LOG = 4_000
MAX_CONTEXT_BYTES = 10 * 1024
MAX_ROUTE = 200
MAX_METHOD = 8
MAX_REQUEST_ID = 64
MAX_USER_AGENT = 500
MAX_APP_VERSION = 40
"""The caps on the free-text columns: text over a cap is cut, never refused."""

FINGERPRINT_MESSAGE_CHARS = 200
"""How much of the message the fingerprint reads: two reports that differ only further down
are the same error."""

STATUS_NEW = "new"
STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_RESOLVED = "resolved"
STATUS_IGNORED = "ignored"
CLOSED_STATUSES = (STATUS_RESOLVED, STATUS_IGNORED)
"""A repeat of a closed event reopens it; an acknowledged one stays acknowledged."""

Recorded = Literal["created", "incremented", "reopened"]

DESCRIPTION_KEY = "description"
"""Where a client report keeps what the user typed: in the event's context, under this key."""

REDACTED = "[REDACTED]"
REDACTED_JWT = "[REDACTED_JWT]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"

# ``Authorization: <scheme> <credentials>``: the scheme stays, the credentials go. A value with
# no scheme at all is a credential too. The header name may be quoted (a serialised request).
_AUTHORIZATION_RE = re.compile(
    r"(Authorization[\"']?\s*[:=]\s*[\"']?)((?:[A-Za-z][\w.-]*\s+)?)([^\s\"',;)}]+)",
    re.IGNORECASE,
)
# A JWT: three base64url segments, the first opening with the ``eyJ`` of a JSON header.
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
# The value of anything named like a secret: ``password=hunter2``, ``"token": "a b"``,
# ``client_secret: 'x'``. The name is matched as a substring, so ``access_token`` and ``api_key``
# count, and so does ``monkey``: redacting too much is the safe side. A quoted value goes
# whole, spaces included; an unquoted one ends at whitespace or a separator, but a ``]`` may be
# part of it (``[REDACTED_JWT]``, which the JWT step has just put there).
_NAMED_SECRET_RE = re.compile(
    r"((?:password|secret|token|code|key)[\"']?\s*[:=]\s*)"
    r"(?:\"([^\"]*)\"|'([^']*)'|([^\s\"',;)}&]+))",
    re.IGNORECASE,
)
_SECRET_NAME_RE = re.compile(r"password|secret|token|code|key", re.IGNORECASE)
# ``Bearer <token>`` with no ``Authorization`` before it is a credential all the same. The
# token is eight or more token characters, so "the bearer of bad news" is left alone.
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_SCRUB_PASSES = 4
"""How many times one scrub may repeat its rules until the text stops changing."""
# An e-mail address: it names an account as surely as an id does. The domain needs a dot, so
# ``user@host`` (a shell-style name) is left alone.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_JS_FRAME_RE = re.compile(r"[\w.-]+\.[A-Za-z]{1,8}:\d+")
_PYTHON_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def _redact_named(match: re.Match[str]) -> str:
    prefix, double, single, _bare = match.groups()
    if double is not None:
        return f'{prefix}"{REDACTED}"'
    if single is not None:
        return f"{prefix}'{REDACTED}'"
    return f"{prefix}{REDACTED}"


def _scrub_once(text: str) -> str:
    text = _AUTHORIZATION_RE.sub(rf"\g<1>\g<2>{REDACTED}", text)
    text = _JWT_RE.sub(REDACTED_JWT, text)
    text = _NAMED_SECRET_RE.sub(_redact_named, text)
    text = _BEARER_RE.sub(rf"\g<1>{REDACTED}", text)
    return _EMAIL_RE.sub(REDACTED_EMAIL, text)


def scrub_secrets(text: str) -> str:
    """Remove credentials and addresses from free text: the credentials of an ``Authorization``
    header, a ``Bearer`` token on its own, JWTs, the value of anything named ``password``,
    ``secret``, ``token``, ``code`` or ``key`` (case-insensitive; the name and the value may be
    quoted; they are joined by ``:`` or ``=``), and e-mail addresses.

    The rules repeat until the text stops changing, so a scrubbed text is a fixed point: one
    pass is not always final, because a word glued to a JWT can be taken for the scheme of an
    ``Authorization`` value, and only the next pass takes the rest.
    """
    if not text:
        return text
    for _ in range(_SCRUB_PASSES):
        scrubbed = _scrub_once(text)
        if scrubbed == text:
            return scrubbed
        text = scrubbed
    return text


def scrub_value(value: Any) -> Any:
    """:func:`scrub_secrets` over a JSON-like structure: every string in it, keys included, and
    the whole value of a key that is named like a secret (``{"token": "abc"}`` keeps no
    ``abc``). Returns a new structure; ``value`` itself is left as it was."""
    if isinstance(value, str):
        return scrub_secrets(value)
    if isinstance(value, dict):
        return {
            scrub_value(key): REDACTED if _is_secret_name(key) else scrub_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [scrub_value(item) for item in value]
    return value


def _is_secret_name(key: Any) -> bool:
    return isinstance(key, str) and _SECRET_NAME_RE.search(key) is not None


def route_template(path: str) -> str:
    """``path`` with every UUID replaced by ``{id}``, and without its query string."""
    return _UUID_RE.sub("{id}", path.partition("?")[0])


def _first_script_frame(stack: str) -> str | None:
    """The first ``file.ext:line`` of a JavaScript stack, by file name and without the column.

    ``host.tld:port`` reads the same way, so a match that follows ``//`` or an identifier
    character is where the app was served from, not where it failed, and is passed over.
    """
    for match in _JS_FRAME_RE.finditer(stack):
        before = stack[: match.start()]
        if before.endswith("//") or (before and re.search(r"\w$", before)):
            continue
        return match.group(0)
    return None


def top_frame(stack: str | None) -> str:
    """Where a stack fails: the first ``file.ext:line`` of a JavaScript stack (a URL counts by
    its file name; the column is not kept), else the first Python frame as ``basename:line``,
    else ``""``."""
    if not stack:
        return ""
    script = _first_script_frame(stack)
    if script is not None:
        return script
    python = _PYTHON_FRAME_RE.search(stack)
    if python is not None:
        name = python.group(1).replace("\\", "/").rsplit("/", 1)[-1]
        return f"{name}:{python.group(2)}"
    return ""


def compute_fingerprint(
    *,
    category: str,
    error_type: str,
    message: str,
    stack: str | None,
    route: str | None,
) -> str:
    """The SHA-256 hex digest of ``category|error_type|message[:200]|top_frame|route``, each
    part scrubbed: two events that agree on these five are the same error."""
    parts = (
        scrub_secrets(category),
        scrub_secrets(error_type),
        scrub_secrets(message)[:FINGERPRINT_MESSAGE_CHARS],
        top_frame(scrub_secrets(stack) if stack else None),
        scrub_secrets(route) if route else "",
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def context_bytes(context: Any) -> int:
    """The size of ``context`` once serialised as JSON; anything that is not JSON becomes
    its ``str``."""
    return len(json.dumps(context, default=str, ensure_ascii=False).encode("utf-8"))


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _cut_strings(value: Any, allowance: int) -> Any:
    if isinstance(value, str):
        return value[:allowance]
    if isinstance(value, dict):
        return {
            _cut_strings(key, allowance): _cut_strings(item, allowance)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_cut_strings(item, allowance) for item in value]
    return value


def _fit_context(context: Any) -> dict[str, Any] | None:
    """The scrubbed context as a JSON object of at most :data:`MAX_CONTEXT_BYTES`.

    A context that is not a mapping is wrapped as ``{"value": ...}``. When the serialised form
    is over the cap, the longest string values are cut first: the structure is rebuilt with
    every string cut to one per-string allowance, halved until the whole fits, so short values
    survive whole. A structure that still does not fit with no text left to cut (thousands of
    numbers) is replaced by a preview of its serialised form.
    """
    if context is None:
        return None
    if not isinstance(context, dict):
        context = {"value": context}
    scrubbed: dict[str, Any] = scrub_value(context)
    if not scrubbed:
        return None
    if context_bytes(scrubbed) <= MAX_CONTEXT_BYTES:
        return scrubbed
    allowance = max(map(len, _strings(scrubbed)), default=0)
    while allowance > 0:
        allowance //= 2
        fitted = _cut_strings(scrubbed, allowance)
        if context_bytes(fitted) <= MAX_CONTEXT_BYTES:
            return fitted
    preview = json.dumps(scrubbed, default=str, ensure_ascii=False)[: MAX_CONTEXT_BYTES - 64]
    return {"truncated": preview}


def _optional(text: str | None, cap: int) -> str | None:
    """Scrubbed and cut to ``cap``; ``None`` when there is nothing (or only blanks) to store."""
    if text is None:
        return None
    text = scrub_secrets(text)
    return text[:cap] if text.strip() else None


def record_error_event(
    db: Session,
    *,
    source: str,
    category: str,
    severity: str,
    error_type: str,
    message: str,
    stack: str | None = None,
    route: str | None = None,
    method: str | None = None,
    status_code: int | None = None,
    request_id: str | None = None,
    user_agent: str | None = None,
    app_version: str | None = None,
    context: Any = None,
    console_log: str | None = None,
    now: datetime | None = None,
) -> tuple[ErrorEvent, Recorded]:
    """Record one error event, rolling repeats up by fingerprint.

    Every free-text field is scrubbed of secrets and cut to its cap; an optional text that is
    blank becomes ``None``; ``context`` is scrubbed and fitted to 10 KB by :func:`_fit_context`.
    A fingerprint not seen before inserts a ``new`` row with a count of 1: ``"created"``. A
    repeat adds one to the count, moves ``last_seen_at`` and refreshes the message, stack,
    context and console text from the latest report: ``"incremented"``, or ``"reopened"`` when
    the event was ``resolved`` or ``ignored`` (it goes back to ``new``; ``acknowledged`` stays).

    Two callers may look the same new fingerprint up before either has committed, and both
    insert: the unique constraint turns the second away, and it continues as a repeat of the
    row that won. Only that insert is undone (a savepoint), so whatever the caller had already
    written in its transaction stays.

    Flushes; the caller commits.
    """
    moment = now or utcnow()
    error_type = scrub_secrets(error_type)[:MAX_ERROR_TYPE]
    message = scrub_secrets(message)[:MAX_MESSAGE]
    stack = _optional(stack, MAX_STACK)
    route = _optional(route, MAX_ROUTE)
    method = _optional(method, MAX_METHOD)
    request_id = _optional(request_id, MAX_REQUEST_ID)
    user_agent = _optional(user_agent, MAX_USER_AGENT)
    app_version = _optional(app_version, MAX_APP_VERSION)
    console_log = _optional(console_log, MAX_CONSOLE_LOG)
    context_json = _fit_context(context)
    fingerprint = compute_fingerprint(
        category=category, error_type=error_type, message=message, stack=stack, route=route
    )
    event = _load(db, fingerprint)
    if event is None:
        event = ErrorEvent(
            source=source,
            category=category,
            severity=severity,
            error_type=error_type,
            message=message,
            fingerprint=fingerprint,
            stack=stack,
            route=route,
            method=method,
            status_code=status_code,
            request_id=request_id,
            user_agent=user_agent,
            app_version=app_version,
            status=STATUS_NEW,
            occurrence_count=1,
            first_seen_at=moment,
            last_seen_at=moment,
            context_json=context_json,
            console_log=console_log,
        )
        try:
            with db.begin_nested():
                db.add(event)
                db.flush()
        except IntegrityError:
            # Another caller inserted this fingerprint between the lookup and the flush: its
            # row won, and this insert alone is rolled back. What follows is a repeat of it.
            event = _load(db, fingerprint)
            if event is None:
                raise
        else:
            db.flush()
            return event, "created"
    event.occurrence_count += 1
    event.last_seen_at = moment
    event.message = message
    event.stack = stack
    event.context_json = context_json
    event.console_log = console_log
    outcome: Recorded = "incremented"
    if event.status in CLOSED_STATUSES:
        event.status = STATUS_NEW
        outcome = "reopened"
    db.flush()
    return event, outcome


def _load(db: Session, fingerprint: str) -> ErrorEvent | None:
    """The stored event with this fingerprint, if there is one."""
    return db.scalar(select(ErrorEvent).where(ErrorEvent.fingerprint == fingerprint))
