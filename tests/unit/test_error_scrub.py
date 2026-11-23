"""The pure rules of the error-events service: scrubbing, route templates, top frames and
fingerprints."""

import pytest

from curvestudio_api.modules.errors.service import (
    compute_fingerprint,
    route_template,
    scrub_secrets,
    scrub_value,
    top_frame,
)

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
BEARER = "Authorization: Bearer s3cr3t-access-token-value"
PASSWORD = "password=hunter2-secret"
UUID = "3f2a9c1e-7b4d-4e8a-9c21-5d6f7a8b9c0d"

# -- scrub_secrets: the three patterns -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Authorization: Bearer abc.def.ghi", "Authorization: Bearer [REDACTED]"),
        ("Authorization: Basic dXNlcjpwYXNz", "Authorization: Basic [REDACTED]"),
        ('"Authorization": "Bearer abc.def.ghi"', '"Authorization": "Bearer [REDACTED]"'),
        ("authorization=Bearer xyz123", "authorization=Bearer [REDACTED]"),
        # a credential with no scheme is still a credential
        ("Authorization: raw-credential", "Authorization: [REDACTED]"),
        # the rest of the line survives
        (f"fetch({BEARER}) failed", "fetch(Authorization: Bearer [REDACTED]) failed"),
    ],
)
def test_an_authorization_header_loses_only_its_credentials(text: str, expected: str) -> None:
    assert scrub_secrets(text) == expected


def test_a_jwt_is_replaced_whole_wherever_it_sits() -> None:
    assert scrub_secrets(f"fetch failed for {JWT} at 12:00") == (
        "fetch failed for [REDACTED_JWT] at 12:00"
    )
    assert scrub_secrets(f"Authorization: Bearer {JWT}") == "Authorization: Bearer [REDACTED]"


def test_a_jwt_in_an_authorization_header_is_redacted_even_after_the_header_pattern() -> None:
    """The header pattern runs first and already removes the token; the result holds neither."""
    assert JWT not in scrub_secrets(f"Authorization: Bearer {JWT}")


@pytest.mark.parametrize("name", ["password", "secret", "token", "code", "key"])
@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("{name}=hunter2", "{name}=[REDACTED]"),
        ("{name}: hunter2", "{name}: [REDACTED]"),
        ('"{name}": "hunter2"', '"{name}": "[REDACTED]"'),
        ("'{name}': 'two word secret'", "'{name}': '[REDACTED]'"),
        ("{NAME} = Hunter2!", "{NAME} = [REDACTED]"),
    ],
)
def test_the_value_of_anything_named_like_a_secret_is_redacted(
    name: str, form: str, expected: str
) -> None:
    text = form.format(name=name, NAME=name.upper())
    assert scrub_secrets(text) == expected.format(name=name, NAME=name.upper())


def test_compound_names_count_and_a_quoted_value_goes_whole() -> None:
    assert scrub_secrets("access_token=abc api_key: def client_secret='g h'") == (
        "access_token=[REDACTED] api_key: [REDACTED] client_secret='[REDACTED]'"
    )
    assert scrub_secrets('"password": "correct horse battery staple"') == (
        '"password": "[REDACTED]"'
    )
    assert scrub_secrets('{"password": "hunter2", "user": "bob"}') == (
        '{"password": "[REDACTED]", "user": "bob"}'
    )


def test_a_value_named_something_else_survives() -> None:
    # No name here is a secret name. An address would go whatever its name: that is the
    # rule ``test_scrub_secrets_removes_email_addresses`` covers.
    text = "username=bob email=bob color=red limit: 10 keyboard: yes"
    assert scrub_secrets(text) == text


def test_every_secret_in_one_text_is_gone() -> None:
    text = f"{BEARER} {PASSWORD} token={JWT} unchanged=yes"
    assert scrub_secrets(text) == (
        "Authorization: Bearer [REDACTED] password=[REDACTED] token=[REDACTED] unchanged=yes"
    )


# -- scrub_value -----------------------------------------------------------------------------------


def test_scrub_value_recurses_through_nested_dicts_and_lists() -> None:
    value = {
        "password": "hunter2",  # the value under a secret-named key
        "detail": f"token={JWT}",  # a string with a secret inside
        "list": [1, BEARER, {"code": "1234", "note": "fine"}, [[PASSWORD]]],
        "api_key": "k",
        "user": "bob",
        "count": 3,
        "flag": True,
        "missing": None,
    }
    assert scrub_value(value) == {
        "password": "[REDACTED]",
        "detail": "token=[REDACTED]",
        "list": [
            1,
            "Authorization: Bearer [REDACTED]",
            {"code": "[REDACTED]", "note": "fine"},
            [["password=[REDACTED]"]],
        ],
        "api_key": "[REDACTED]",
        "user": "bob",
        "count": 3,
        "flag": True,
        "missing": None,
    }


def test_scrub_value_rewrites_keys_and_passes_scalars_through() -> None:
    # the key is scrubbed as a string; its value survives because the key names no secret
    assert scrub_value({"Authorization: Bearer abc": 1, "plain": 2}) == {
        "Authorization: Bearer [REDACTED]": 1,
        "plain": 2,
    }
    # a key that so much as mentions a secret name loses its value: the safe side
    assert scrub_value({"status_code": 500, "api_key": "k"}) == {
        "status_code": "[REDACTED]",
        "api_key": "[REDACTED]",
    }
    assert scrub_value(BEARER) == "Authorization: Bearer [REDACTED]"
    assert scrub_value(5) == 5
    assert scrub_value(None) is None


def test_scrub_value_does_not_change_the_value_it_was_given() -> None:
    value: dict[str, object] = {"password": "hunter2", "inner": {"token": "t"}}
    scrubbed = scrub_value(value)
    assert scrubbed != value
    assert value == {"password": "hunter2", "inner": {"token": "t"}}


# -- route_template --------------------------------------------------------------------------------


def test_route_template_replaces_every_uuid_and_drops_the_query_string() -> None:
    assert route_template(f"/datasets/{UUID}") == "/datasets/{id}"
    assert route_template(f"/datasets/{UUID}/fits/{UUID}") == "/datasets/{id}/fits/{id}"
    assert route_template("/api/v1/datasets") == "/api/v1/datasets"
    assert route_template("/api/v1/datasets/{dataset_id}") == "/api/v1/datasets/{dataset_id}"
    assert route_template(f"/datasets/{UUID}?tab=fits&x={UUID}") == "/datasets/{id}"
    assert route_template("") == ""


# -- top_frame -------------------------------------------------------------------------------------


def test_top_frame_takes_the_first_javascript_frame() -> None:
    stack = (
        "TypeError: Cannot read properties of undefined (reading 'x')\n"
        "    at render (https://app.example.org/assets/index-abc123.js:1:2345)\n"
        "    at save (app.js:12:5)\n"
    )
    assert top_frame(stack) == "index-abc123.js:1"
    assert top_frame("Error: boom\n    at save (app.js:12:5)") == "app.js:12"


def test_top_frame_takes_the_first_python_frame_by_basename() -> None:
    stack = (
        "Traceback (most recent call last):\n"
        '  File "/srv/app/curvestudio_api/jobs.py", line 265, in _run\n'
        '    File "C:\\srv\\app\\worker.py", line 10, in run\n'
        "RuntimeError: boom\n"
    )
    assert top_frame(stack) == "jobs.py:265"
    assert top_frame(stack.replace("jobs.py", "other.py")) == "other.py:265"


def test_top_frame_is_empty_without_a_frame() -> None:
    assert top_frame("TypeError: x is undefined") == ""
    assert top_frame("Error at 12:30:00 for file.json: see log") == ""
    assert top_frame("version 1.2:3") == ""
    assert top_frame(None) == ""
    assert top_frame("") == ""


# -- compute_fingerprint ---------------------------------------------------------------------------


BASE = {
    "category": "crash",
    "error_type": "TypeError",
    "message": "boom",
    "stack": "Error: boom\n    at save (app.js:12:5)",
    "route": "/datasets/{id}",
}


def test_the_fingerprint_is_a_sha256_hex_digest_that_is_stable() -> None:
    value = compute_fingerprint(**BASE)  # type: ignore[arg-type]
    assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    assert compute_fingerprint(**BASE) == value  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("part", "other"),
    [
        ("category", "ui"),
        ("error_type", "RangeError"),
        ("message", "bang"),
        ("stack", "Error: boom\n    at save (app.js:99:5)"),
        ("route", "/fits/{id}"),
    ],
)
def test_the_fingerprint_changes_with_each_of_its_five_parts(part: str, other: str) -> None:
    changed = {**BASE, part: other}
    assert compute_fingerprint(**changed) != compute_fingerprint(**BASE)  # type: ignore[arg-type]


def test_the_fingerprint_ignores_the_message_beyond_its_first_200_characters() -> None:
    tail = "x" * 100
    long_message = "b" * 250 + tail
    original = compute_fingerprint(**{**BASE, "message": long_message})  # type: ignore[arg-type]
    same_tail = compute_fingerprint(**{**BASE, "message": "b" * 250 + "y" * 100})  # type: ignore[arg-type]
    same_length = compute_fingerprint(**{**BASE, "message": "b" * 250})  # type: ignore[arg-type]
    assert original == same_tail
    assert original == same_length
    assert compute_fingerprint(**{**BASE, "message": "b" * 199 + "z" + tail}) != original  # type: ignore[arg-type]


def test_the_fingerprint_is_computed_on_scrubbed_text() -> None:
    assert compute_fingerprint(**{**BASE, "message": f"password={JWT}"}) == compute_fingerprint(  # type: ignore[arg-type]
        **{**BASE, "message": "password=[REDACTED]"}
    )
    assert compute_fingerprint(**{**BASE, "route": None}) == compute_fingerprint(  # type: ignore[arg-type]
        **{**BASE, "route": ""}
    )


def test_scrub_secrets_removes_email_addresses() -> None:
    """An address identifies an account as surely as an id does, so none may be stored."""
    assert scrub_secrets('no account for "email": "bob@example.org"') == (
        'no account for "email": "[REDACTED_EMAIL]"'
    )
    assert scrub_secrets("smtp said: recipient a.b+c@sub.example.co.in rejected") == (
        "smtp said: recipient [REDACTED_EMAIL] rejected"
    )
    # not an address: the domain has no dot
    assert scrub_secrets("see user@host for details") == "see user@host for details"


def test_the_fingerprint_does_not_depend_on_the_address_in_a_message() -> None:
    first = compute_fingerprint(**{**BASE, "message": "no account bob@example.org"})
    second = compute_fingerprint(**{**BASE, "message": "no account carol@example.org"})  # type: ignore[arg-type]
    assert first == second  # type: ignore[arg-type]


def test_top_frame_skips_a_host_and_port() -> None:
    """A dotted host with a port reads like ``name.tld:3000``. It is where the app was served
    from, not where it failed: every error would otherwise share one top frame."""
    assert top_frame("at f (http://dev.local:3000/assets/app.js:1:2)") == "app.js:1"
    assert top_frame("at f (webpack://app/./src/x.js:1:1)") == "x.js:1"
    assert top_frame("at f (https://app.example/assets/index-abc123.js:1:2345)") == (
        "index-abc123.js:1"
    )


def test_scrub_secrets_is_idempotent() -> None:
    """A second scrub never changes a scrubbed text. One pass is not always final: a word glued
    to a JWT can be taken for the scheme of an ``Authorization`` value, and only the next pass
    takes the rest."""
    glued = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sigpart"
    for text in (
        f"Authorization:client_secret{glued}\t/ = ",
        f"Authorization: Bearer {JWT}",
        f"{BEARER} {PASSWORD} token={JWT} unchanged=yes",
        'no account "email": "bob@example.org" password=hunter2',
    ):
        once = scrub_secrets(text)
        assert scrub_secrets(once) == once


def test_scrub_secrets_removes_a_bare_bearer_token() -> None:
    """``Bearer <token>`` with no ``Authorization`` before it is still a credential."""
    assert scrub_secrets("got Bearer abc123.def456-ghi back") == "got Bearer [REDACTED] back"
    assert scrub_secrets(f"sent bearer {JWT} out") == "sent bearer [REDACTED_JWT] out"
    # a short word after "bearer" is a word, and an already redacted value stays as it is
    assert scrub_secrets("the bearer of bad news") == "the bearer of bad news"
    assert scrub_secrets("Bearer [REDACTED]") == "Bearer [REDACTED]"
