"""Transactional email delivery via Brevo.

Provider-neutral at the call site: callers get a bool and never an exception --
a provider outage must not turn a route into a 500. But it is loudly
fail-VISIBLE rather than fail-silent: every failure path print()s an ERROR line
before returning False, and the caller is expected to surface a 503 rather than
a cheerful {"status": "ok"} for a message nobody will receive.

print(), not logging: dashboard.backend.* loggers sit at WARNING in every real
deployment (nothing configures logging, and uvicorn's LOGGING_CONFIG has no
'root' key), so logger.error() here would be invisible exactly where it matters.

Config is read at call time, not import time, so tests can set and unset freely.
"""

import os

import httpx

BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
SEND_TIMEOUT_SECONDS = 10.0
DEFAULT_FROM_NAME = "Agentic Trading Lab"
_PROVIDER_ERROR_SNIPPET_CHARS = 200


def email_configured() -> bool:
    """True when the provider credentials needed to send anything are present.

    Public and called from send_email() below (not just tested standalone) so
    the "are we configured" check exists in exactly one place.
    """
    return bool(os.getenv("BREVO_API_KEY") and os.getenv("ACCOUNT_EMAIL_FROM"))


def _one_line(value: str) -> str:
    """Collapse a provider-supplied string to a single log line.

    The response body can echo submitted content, so an embedded newline would
    let it forge a second log entry. Log-injection defence only -- nothing here
    is a security boundary.
    """
    return value.replace("\r", " ").replace("\n", " ")


async def send_email(to: str, subject: str, text_body: str) -> bool:
    """Send one plain-text transactional email. True on a 2xx, False otherwise."""
    if not email_configured():
        print(
            "ERROR: transactional email requested but BREVO_API_KEY / "
            "ACCOUNT_EMAIL_FROM are not set -- no message was sent"
        )
        return False
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("ACCOUNT_EMAIL_FROM")

    payload = {
        "sender": {
            "name": os.getenv("ACCOUNT_EMAIL_FROM_NAME") or DEFAULT_FROM_NAME,
            "email": sender_email,
        },
        "to": [{"email": to}],
        "subject": subject,
        "textContent": text_body,
    }

    try:
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            response = await client.post(
                BREVO_ENDPOINT,
                headers={"api-key": api_key, "accept": "application/json"},
                json=payload,
            )
    except Exception as exc:  # noqa: BLE001 -- a provider outage must not 500 the route
        print(f"ERROR: transactional email POST failed: {exc!r}")
        return False

    if response.status_code // 100 != 2:
        snippet = _one_line(response.text[:_PROVIDER_ERROR_SNIPPET_CHARS])
        print(
            f"ERROR: transactional email rejected by provider "
            f"(HTTP {response.status_code}): {snippet}"
        )
        return False
    return True
