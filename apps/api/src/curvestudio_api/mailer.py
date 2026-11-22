"""Branded transactional e-mail: verification and password-reset links.

The shell is the shared ci-app-mail template: one accent colour (``#e5461f``), a 560 px
table layout, inline styles, a CID-embedded logo and social icons, and a plain-text
alternative. Callers hand over a plain-text body; :func:`render_email` builds both parts.

``send_email`` is the only path that talks to SMTP. It reads its own :class:`Settings`
from the environment (the app's settings come from there too), so the router only needs
the request settings to build the links.
"""

import html
import logging
import re
import smtplib
from datetime import UTC, datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from functools import cache
from importlib import resources

from curvestudio_api.settings import Settings

logger = logging.getLogger(__name__)

APP_NAME = "CurveStudio"
COMPANY_URL = "https://www.concreteinfo.in"
SUPPORT_EMAIL = "support@concreteinfo.in"
LINKEDIN_URL = "https://in.linkedin.com/company/concreteinfo"
YOUTUBE_URL = "https://www.youtube.com/@concreteinfo6736"

_ACCENT = "#e5461f"
_LOGO_WIDTH, _LOGO_HEIGHT = 150, 42
_URL_RE = re.compile(r'(https?://[^\s<>"\']+)', re.IGNORECASE)
_PAGE_TABLE_STYLE = "background-color:#f1f1f1;"
_SHELL_STYLE = (
    "max-width:560px;width:100%;background-color:#ffffff;"
    "border:1px solid #e3e3e3;border-radius:8px;"
)
_RULE_STYLE = (
    f"padding:0 32px 20px 32px;border-bottom:3px solid {_ACCENT};font-size:0;line-height:0;"
)
_BODY_STYLE = (
    "padding:24px 32px 8px 32px;color:#1c1917;font-size:15px;line-height:1.6;"
    "font-family:'Segoe UI',Arial,Helvetica,sans-serif;"
)


def _asset(name: str) -> bytes | None:
    """A bundled PNG, or ``None`` so a missing asset never blocks a send."""
    try:
        return (
            resources.files("curvestudio_api") / "assets" / "email" / f"{name}.png"
        ).read_bytes()
    except FileNotFoundError, OSError:
        return None


@cache
def get_logo_bytes() -> bytes | None:
    return _asset("logo")


@cache
def get_social_icons() -> dict[str, bytes]:
    icons: dict[str, bytes] = {}
    for name in ("icon-linkedin", "icon-youtube"):
        png = _asset(name)
        if png is not None:
            icons[name] = png
    return icons


def _linkify(escaped: str) -> str:
    """Turn bare URLs into accent-coloured links; trailing punctuation stays out."""

    def repl(match: re.Match[str]) -> str:
        url = match.group(1)
        trailing = ""
        while url and url[-1] in ".,);":
            trailing = url[-1] + trailing
            url = url[:-1]
        return (
            f'<a href="{url}" style="color:{_ACCENT};text-decoration:underline;">'
            f"{url}</a>{trailing}"
        )

    return _URL_RE.sub(repl, escaped)


def _paragraphs(body_text: str) -> str:
    """Escape and linkify the body, split on blank lines into styled ``<p>`` blocks."""
    parts = []
    for chunk in re.split(r"\n\s*\n", body_text.strip()):
        escaped = html.escape(chunk).replace("\n", "<br>")
        parts.append(f'<p style="margin:0 0 16px 0;">{_linkify(escaped)}</p>')
    return "".join(parts)


def _footer_text(app_url: str) -> str:
    return (
        f"—\n{APP_NAME}: {app_url}\n"
        f"ConcreteInfo: {COMPANY_URL}\n"
        f"Support: {SUPPORT_EMAIL}\n"
        f"LinkedIn: {LINKEDIN_URL} | YouTube: {YOUTUBE_URL}"
    )


def _social_row() -> str:
    """Monochrome brand-palette icons (grey chips), linked to the social profiles."""
    if not get_social_icons():
        return ""
    icon = (
        '<a href="{url}" style="text-decoration:none;">'
        '<img src="cid:{cid}" width="20" height="20" alt="{alt}" '
        'style="border:0;vertical-align:middle;"></a>'
    )
    return (
        '<p style="margin:10px 0 8px 0;">'
        + icon.format(url=LINKEDIN_URL, cid="icon-linkedin", alt="ConcreteInfo on LinkedIn")
        + " &nbsp; "
        + icon.format(url=YOUTUBE_URL, cid="icon-youtube", alt="ConcreteInfo on YouTube")
        + "</p>"
    )


def _footer_html(app_url: str) -> str:
    return (
        '<tr><td style="padding:20px 32px 28px 32px;border-top:1px solid #e3e3e3;'
        'color:#464646;font-size:12px;line-height:1.8;">'
        f'<a href="{app_url}" style="color:{_ACCENT};text-decoration:none;">'
        f"{html.escape(APP_NAME)}</a><br>"
        f'<a href="{COMPANY_URL}" style="color:#464646;text-decoration:underline;">'
        "ConcreteInfo — www.concreteinfo.in</a> &nbsp;·&nbsp; "
        f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#464646;'
        f'text-decoration:underline;">Support: {SUPPORT_EMAIL}</a>'
        + _social_row()
        + f"<br>© {datetime.now(UTC).year} ConcreteInfo"
        "</td></tr>"
    )


def render_email(settings: Settings, subject: str, body_text: str) -> tuple[str, str]:
    """Return ``(plain_text, html)`` for a plain-text body under the branded shell."""
    app_url = settings.app_url.rstrip("/")
    plain = f"{body_text.rstrip()}\n\n{_footer_text(app_url)}"

    logo_img = (
        f'<a href="{app_url}"><img src="cid:logo" width="{_LOGO_WIDTH}" '
        f'height="{_LOGO_HEIGHT}" alt="{html.escape(APP_NAME)}" '
        'style="display:block;border:0;max-width:100%;height:auto;"></a>'
    )

    html_part = f"""<!doctype html>
<html lang="en">
<body style="margin:0;padding:0;background-color:#f1f1f1;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="{_PAGE_TABLE_STYLE}">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" cellpadding="0" cellspacing="0" style="{_SHELL_STYLE}">
<tr><td style="padding:28px 32px 12px 32px;">{logo_img}</td></tr>
<tr><td style="{_RULE_STYLE}">&nbsp;</td></tr>
<tr><td style="{_BODY_STYLE}">
{_paragraphs(body_text)}
</td></tr>
{_footer_html(app_url)}
</table>
</td></tr></table>
</body></html>"""
    return plain, html_part


def _build_message(settings: Settings, to: str, subject: str, body: str) -> MIMEMultipart:
    plain, html_part = render_email(settings, subject, body)

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = formataddr((settings.email_from_name, settings.smtp_from or settings.smtp_user))
    msg["To"] = to

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(plain, "plain", "utf-8"))
    alternative.attach(MIMEText(html_part, "html", "utf-8"))
    msg.attach(alternative)

    for cid, png in {"logo": get_logo_bytes(), **get_social_icons()}.items():
        if not png:
            continue
        image = MIMEImage(png, _subtype="png")
        image.add_header("Content-ID", f"<{cid}>")
        image.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(image)
    return msg


def send_email(to: str, subject: str, body: str) -> bool:
    """Send one branded e-mail. Returns ``True`` when it was handed to SMTP.

    With no ``smtp_host`` configured the send is skipped (logged) and ``False`` returned;
    an SMTP failure is logged with its traceback, never swallowed silently.
    """
    settings = Settings()
    if not settings.smtp_host:
        logger.info("SMTP not configured; email to %s skipped: %s", to, subject)
        return False
    try:
        message = _build_message(settings, to, subject, body)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            server.ehlo()
            if settings.smtp_port == 587:
                server.starttls()
                server.ehlo()
            if settings.smtp_user:
                password = (
                    settings.smtp_password.get_secret_value() if settings.smtp_password else ""
                )
                server.login(settings.smtp_user, password)
            server.send_message(message)
    except Exception:
        logger.exception("sending email to %s failed: %s", to, subject)
        return False
    return True


def _link(settings: Settings, path: str, token: str) -> str:
    return f"{settings.app_url.rstrip('/')}/{path}?token={token}"


def send_verification_email(user_email: str, token: str, *, settings: Settings) -> bool:
    """The "confirm your address" mail; its link opens ``/verify-email``."""
    url = _link(settings, "verify-email", token)
    subject = f"Verify your email address for {APP_NAME}"
    body = (
        f"Welcome to {APP_NAME}.\n\n"
        "Confirm this email address to finish setting up your account:\n\n"
        f"{url}\n\n"
        f"This link expires in {settings.verification_ttl_hours} hours. "
        "If you did not create an account, you can ignore this email.\n\n"
        f"— The {APP_NAME} team"
    )
    if settings.env == "test":
        logger.info("verification link for %s: %s", user_email, url)
    return send_email(user_email, subject, body)


def send_password_reset_email(user_email: str, token: str, *, settings: Settings) -> bool:
    """The "choose a new password" mail; its link opens ``/reset-password``."""
    url = _link(settings, "reset-password", token)
    subject = f"Reset your {APP_NAME} password"
    body = (
        f"We received a request to reset the password for your {APP_NAME} account.\n\n"
        "Choose a new password here:\n\n"
        f"{url}\n\n"
        f"This link expires in {settings.reset_ttl_minutes} minutes. "
        "If you did not ask for a reset, no action is needed and your password stays the same.\n\n"
        f"— The {APP_NAME} team"
    )
    if settings.env == "test":
        logger.info("password reset link for %s: %s", user_email, url)
    return send_email(user_email, subject, body)
