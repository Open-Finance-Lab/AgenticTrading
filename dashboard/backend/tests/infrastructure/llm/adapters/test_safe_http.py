"""Outbound provider verification must stay on public, pinned addresses."""

from __future__ import annotations

import socket

import pytest

from dashboard.backend.infrastructure.llm.adapters.safe_http import (
    build_pinned_transport,
    build_explicit_proxy_transport,
    PinnedNetworkBackend,
    ResolvedAddress,
    UnsafeProviderAddress,
    resolve_public_addresses,
)


@pytest.mark.parametrize(
    "sockaddr",
    [
        ("127.0.0.1", 443),
        ("10.0.0.8", 443),
        ("169.254.169.254", 443),
        ("192.0.2.10", 443),
    ],
)
def test_dns_results_in_private_link_local_or_reserved_ranges_are_rejected(
    monkeypatch, sockaddr
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)
        ],
    )

    with pytest.raises(UnsafeProviderAddress):
        resolve_public_addresses("provider.example", 443)


def test_dns_results_are_deduplicated_and_preserve_family(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443)),
        ],
    )

    addresses = resolve_public_addresses("provider.example", 443)

    assert len(addresses) == 1
    assert addresses[0].family == socket.AF_INET
    assert addresses[0].ip == "93.184.216.34"
    assert addresses[0].sockaddr == ("93.184.216.34", 443)


def test_metadata_hostnames_are_rejected_without_dns(monkeypatch):
    called = False

    def fail_if_resolved(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("metadata host must be rejected before DNS")

    monkeypatch.setattr(socket, "getaddrinfo", fail_if_resolved)

    with pytest.raises(UnsafeProviderAddress):
        resolve_public_addresses("metadata.google.internal", 443)

    assert called is False


def test_pinned_backend_connects_to_validated_ip_without_second_dns_lookup(monkeypatch):
    calls = []

    class FakeSocket:
        def __init__(self, family, _kind):
            self.family = family

        def settimeout(self, value):
            calls.append(("timeout", value))

        def setsockopt(self, *option):
            calls.append(("option", option))

        def connect(self, sockaddr):
            calls.append(("connect", sockaddr))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(socket, "socket", FakeSocket)
    backend = PinnedNetworkBackend(
        {
            ("provider.example", 443): (
                ResolvedAddress(
                    family=socket.AF_INET,
                    ip="93.184.216.34",
                    sockaddr=("93.184.216.34", 443),
                ),
            )
        }
    )

    backend.connect_tcp("provider.example", 443, timeout=3.0)

    assert ("timeout", 3.0) in calls
    assert ("connect", ("93.184.216.34", 443)) in calls
    assert not any(item[0] == "connect" and "provider.example" in item[1] for item in calls)


@pytest.mark.parametrize(
    "url",
    [
        "http://provider.example/v1/models",
        "https://user:password@provider.example/v1/models",
        "https://provider.example/v1/models?token=fake",
    ],
)
def test_pinned_transport_rejects_non_origin_url(url):
    with pytest.raises(UnsafeProviderAddress):
        build_pinned_transport(url)


def test_pinned_transport_does_not_use_environment_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        ],
    )

    transport = build_pinned_transport("https://provider.example/v1/models")
    try:
        assert type(transport._transport._pool).__name__ == "ConnectionPool"
        assert isinstance(transport._transport._pool._network_backend, PinnedNetworkBackend)
    finally:
        transport.close()


def test_explicit_proxy_transport_is_opt_in_and_does_not_read_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    transport = build_explicit_proxy_transport("http://127.0.0.1:7897")
    try:
        assert type(transport).__name__ == "HTTPTransport"
        assert transport._pool._proxy_url is not None
    finally:
        transport.close()


@pytest.mark.parametrize(
    "proxy_url",
    [
        "",
        "ftp://127.0.0.1:7897",
        "http://user:pass@127.0.0.1:7897",
        "http://127.0.0.1:7897/?token=secret",
    ],
)
def test_explicit_proxy_transport_rejects_unsafe_proxy_url(proxy_url):
    with pytest.raises(UnsafeProviderAddress):
        build_explicit_proxy_transport(proxy_url)
