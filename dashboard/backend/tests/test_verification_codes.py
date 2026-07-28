"""Short-lived confirmation codes for the email-change flow."""

from dashboard.backend.verification_codes import (
    CODE_ALPHABET,
    CODE_LENGTH,
    generate_code,
    hash_code,
)


def test_generated_code_has_the_expected_shape():
    for _ in range(50):
        code = generate_code()
        assert len(code) == CODE_LENGTH == 6
        assert set(code) <= set(CODE_ALPHABET)


def test_alphabet_excludes_the_characters_users_misread():
    # 0/O and 1/I/L are the pairs users transcribe wrong off a phone screen.
    for ambiguous in "0O1IL":
        assert ambiguous not in CODE_ALPHABET
    assert len(CODE_ALPHABET) == 31
    assert len(set(CODE_ALPHABET)) == 31


def test_generated_codes_are_not_all_identical():
    assert len({generate_code() for _ in range(50)}) > 1


def test_hash_is_case_insensitive_and_whitespace_tolerant():
    assert hash_code("abc234") == hash_code("ABC234")
    assert hash_code("  ABC234  ") == hash_code("ABC234")


def test_hash_is_stable_and_distinguishes_codes():
    assert hash_code("ABC234") == hash_code("ABC234")
    assert hash_code("ABC234") != hash_code("ABC235")
    assert len(hash_code("ABC234")) == 64  # sha256 hex
