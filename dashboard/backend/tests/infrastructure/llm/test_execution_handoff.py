import base64
import io
import sys

import pytest

from dashboard.backend.infrastructure.llm.execution.handoff import (
    _MAX_HANDOFF_BYTES,
    ExecutionHandoffError,
    HandoffReplayGuard,
    consume_execution_handoff,
    create_execution_handoff,
    read_execution_handoff,
)


def _token(now=1_000):
    return create_execution_handoff(
        user_id=1,
        run_id="run-1",
        billing_mode="platform_credits",
        provider_id="openrouter",
        model_id="qwen/qwen3.7-plus",
        now=now,
    )


@pytest.mark.parametrize(
    "token",
    ["é.abc", "\udcff.abc", "abc.é", "", "abc", ".abc", "abc.", "a.b.c", "abc=.abc"],
    ids=[
        "non-ascii-payload",
        # What a corrupt stdin byte becomes under C.UTF-8 (surrogateescape).
        "lone-surrogate",
        "non-ascii-signature",
        "empty",
        "no-separator",
        "empty-payload",
        "empty-signature",
        "two-separators",
        "padding",
    ],
)
def test_a_malformed_token_is_a_handoff_error(token):
    # The worker catches only ExecutionHandoffError; a raw UnicodeEncodeError
    # from the payload half reached it as a traceback (#659).
    with pytest.raises(ExecutionHandoffError):
        consume_execution_handoff(token, now=1_001)


def _signature_variants(token):
    payload, signature = token.split(".")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    # A 32-byte signature is 43 characters carrying 258 bits, so the low two
    # bits of the last character are spare: flipping one changes the string,
    # not the bytes it decodes to.
    spare_bit = alphabet[alphabet.index(signature[-1]) ^ 1]
    return {
        "junk": f"{payload}.{signature[:10]}!!{signature[10:]}",
        "spaces": f"{payload}.{signature[:10]}  {signature[10:]}",
        "control": f"{payload}.{signature[:10]}\n\t{signature[10:]}",
        "spare-bit": f"{payload}.{signature[:-1]}{spare_bit}",
    }


@pytest.mark.parametrize("variant", ["junk", "spaces", "control", "spare-bit"])
def test_a_signature_that_only_decodes_like_the_real_one_is_refused(variant):
    token = _token()
    altered = _signature_variants(token)[variant]
    assert base64.urlsafe_b64decode(altered.split(".")[1] + "=") == (
        base64.urlsafe_b64decode(token.split(".")[1] + "=")
    ), "the variant must decode to the genuine signature bytes"

    with pytest.raises(ExecutionHandoffError):
        consume_execution_handoff(altered, replay_guard=HandoffReplayGuard(), now=1_001)


def test_a_second_claim_in_one_process_is_refused():
    token = _token()
    guard = HandoffReplayGuard()
    consume_execution_handoff(token, replay_guard=guard, now=1_001)

    with pytest.raises(ExecutionHandoffError):
        consume_execution_handoff(token, replay_guard=guard, now=1_002)


@pytest.mark.parametrize("framing", ["", "\n", "\r\n", " \n"])
def test_read_strips_stream_framing_around_a_token(framing):
    # `echo "$TOKEN" |` appends a newline; the token itself never has one.
    token = _token()
    stream = io.BytesIO(f"{token}{framing}".encode("ascii"))

    handoff = consume_execution_handoff(
        read_execution_handoff(stream), replay_guard=HandoffReplayGuard(), now=1_001
    )

    assert handoff.run_id == "run-1"


def test_read_refuses_bytes_that_are_not_ascii():
    with pytest.raises(ExecutionHandoffError):
        read_execution_handoff(io.BytesIO(b"\xff.abc"))


def test_read_stops_one_byte_past_the_limit():
    stream = io.BytesIO(b"a" * (_MAX_HANDOFF_BYTES * 4))

    with pytest.raises(ExecutionHandoffError):
        read_execution_handoff(stream)

    assert stream.tell() == _MAX_HANDOFF_BYTES + 1


@pytest.mark.parametrize("errors", ["strict", "surrogateescape"])
def test_worker_refuses_a_corrupt_stdin_byte_cleanly(monkeypatch, capsys, errors):
    # `strict` is a dev shell's en_US.UTF-8; `surrogateescape` is the image's
    # C.UTF-8. Before the worker read bytes, strict raised UnicodeDecodeError
    # from sys.stdin.read() itself, outside the handoff error the worker catches.
    from dashboard.scripts import backtest_hourly_agent as cli

    stdin = io.TextIOWrapper(io.BytesIO(b"\xff.abc"), encoding="utf-8", errors=errors)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        sys,
        "argv",
        ["backtest_hourly_agent", "--execution-handoff-stdin", "--run-id", "run-1"],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "invalid execution handoff" in capsys.readouterr().err


def test_handoff_round_trips_ordered_provider_candidates():
    payload = create_execution_handoff(
        user_id=1,
        run_id="run-1",
        billing_mode="platform_credits",
        provider_id="openrouter",
        provider_ids=("openrouter", "commonstack"),
        model_id="qwen/qwen3.7-plus",
        now=1_000,
    )

    handoff = consume_execution_handoff(payload, now=1_001)

    assert handoff.provider_id == "openrouter"
    assert handoff.provider_ids == ("openrouter", "commonstack")


def test_handoff_rejects_duplicate_or_misordered_candidates():
    with pytest.raises(ValueError):
        create_execution_handoff(
            user_id=1,
            run_id="run-1",
            billing_mode="platform_credits",
            provider_id="openrouter",
            provider_ids=("commonstack", "openrouter"),
            model_id="qwen/qwen3.7-plus",
        )

    with pytest.raises(ValueError):
        create_execution_handoff(
            user_id=1,
            run_id="run-1",
            billing_mode="platform_credits",
            provider_id="openrouter",
            provider_ids=("openrouter", "openrouter"),
            model_id="qwen/qwen3.7-plus",
        )
