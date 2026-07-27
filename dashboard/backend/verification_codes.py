"""Short-lived alphanumeric confirmation codes.

Used by the email-change flow; the password-reset flow (#187) is the intended
second consumer, which is why this is a standalone module rather than private
helpers inside api/auth.py.
"""

import hashlib
import secrets

# 31 symbols: digits and uppercase letters, minus 0/O and 1/I/L -- the pairs a
# user misreads off a phone screen and types back wrong. 31**6 is about
# 8.9e8 combinations, which a 5-attempt cap and a 15-minute expiry make
# comfortably unguessable.
CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
CODE_LENGTH = 6


def generate_code() -> str:
    """Return a fresh code drawn from a CSPRNG."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def hash_code(code: str) -> str:
    """SHA-256 hex of the normalized (stripped, uppercased) code.

    Normalizing here is what makes comparison case-insensitive, so a user typing
    lowercase still matches.

    SHA-256 rather than bcrypt: this is a short-lived high-entropy secret, and
    what stops guessing is the attempt cap and the expiry, not hash cost --
    bcrypt would add work to every attempt for no gain. The hash is not an
    offline-attack defence (1e9 candidates is trivially searchable, and anyone
    with database write access could simply rewrite users.email). It stops a
    *casual* read -- a log line, a backup, a support query -- from yielding a
    live code.
    """
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()
