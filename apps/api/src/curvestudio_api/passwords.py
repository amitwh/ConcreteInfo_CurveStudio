"""Password policy beyond length: not the e-mail's local part, not a well-known password."""

from functools import cache
from importlib import resources


@cache
def common_passwords() -> frozenset[str]:
    """The bundled list of common passwords (lower case)."""
    text = (resources.files("curvestudio_api") / "common_passwords.txt").read_text(encoding="utf-8")
    return frozenset(
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    )


def weakness(password: str, email: str) -> str | None:
    """Why ``password`` is refused, or ``None`` when it is acceptable."""
    lowered = password.lower()
    local_part = email.split("@", 1)[0].lower()
    if local_part and lowered == local_part:
        return "the password must not be the same as the part of the e-mail address before the @"
    if lowered in common_passwords():
        return "this password is too common; choose another"
    return None
