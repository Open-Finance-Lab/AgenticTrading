"""Public-address DNS resolution and IP-pinned HTTPS transport."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from typing import Iterable
from urllib.parse import urlsplit

import httpcore
import httpx
from httpcore._backends.sync import SyncStream


class UnsafeProviderAddress(ValueError):
    """The provider hostname resolves to an address outside the public Internet."""


class ProviderAddressResolutionError(OSError):
    """The provider hostname could not be resolved."""


@dataclass(frozen=True)
class ResolvedAddress:
    family: int
    ip: str
    sockaddr: tuple[object, ...]


_BLOCKED_HOSTNAMES = {
    "instance-data",
    "instance-data.ec2.internal",
    "metadata.google.com",
    "metadata.google.internal",
}


def _canonical_host(host: str) -> str:
    value = host.strip().rstrip(".").lower()
    try:
        return value.encode("idna").decode("ascii")
    except UnicodeError:
        return value


def _assert_public_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise UnsafeProviderAddress("provider address is not a public IP") from exc
    # is_global excludes loopback, link-local, RFC1918, documentation, reserved,
    # multicast, and unspecified ranges. This is intentionally fail-closed.
    if not address.is_global:
        raise UnsafeProviderAddress("provider address is not a public IP")
    return address


def _public_sockaddr(
    family: int, ip: str, port: int, original: tuple[object, ...]
) -> tuple[object, ...]:
    if family == socket.AF_INET6:
        scope_id = original[3] if len(original) > 3 else 0
        return (ip, port, 0, scope_id)
    return (ip, port)


def resolve_public_addresses(host: str, port: int) -> tuple[ResolvedAddress, ...]:
    """Resolve a provider once and reject every non-public DNS answer.

    The returned addresses are later used by the pinned network backend. We do
    not resolve again during connect, which prevents DNS rebinding between the
    safety check and the TCP connection.
    """

    canonical = _canonical_host(host)
    if not canonical:
        raise UnsafeProviderAddress("provider hostname is empty")
    if (
        canonical in _BLOCKED_HOSTNAMES
        or canonical.endswith(".internal")
        or canonical.endswith(".local")
    ):
        raise UnsafeProviderAddress("provider hostname is not allowed")

    try:
        literal = ipaddress.ip_address(canonical)
    except ValueError:
        literal = None
    if literal is not None:
        _assert_public_ip(str(literal))
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        sockaddr = _public_sockaddr(family, str(literal), int(port), ())
        return (ResolvedAddress(family=family, ip=str(literal), sockaddr=sockaddr),)

    try:
        records = socket.getaddrinfo(
            canonical,
            int(port),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ProviderAddressResolutionError(
            "provider hostname could not be resolved"
        ) from exc

    addresses: list[ResolvedAddress] = []
    seen: set[tuple[int, tuple[object, ...]]] = set()
    for family, _socktype, _proto, _canonname, sockaddr in records:
        if family not in {socket.AF_INET, socket.AF_INET6} or not sockaddr:
            continue
        ip = str(sockaddr[0])
        _assert_public_ip(ip)
        normalized = _public_sockaddr(family, ip, int(port), tuple(sockaddr))
        key = (family, normalized)
        if key in seen:
            continue
        seen.add(key)
        addresses.append(ResolvedAddress(family=family, ip=ip, sockaddr=normalized))

    if not addresses:
        raise ProviderAddressResolutionError(
            "provider hostname returned no usable addresses"
        )
    return tuple(addresses)


class PinnedNetworkBackend(httpcore.SyncBackend):
    """Connect to prevalidated IPs while retaining the original HTTP origin.

    httpcore still receives the original hostname as its connection origin, so
    HTTP Host and TLS SNI remain the provider hostname. Only the TCP dial target
    is replaced with the validated address list.
    """

    def __init__(
        self, addresses_by_origin: dict[tuple[str, int], Iterable[ResolvedAddress]]
    ):
        self._addresses_by_origin = {
            (_canonical_host(host), int(port)): tuple(addresses)
            for (host, port), addresses in addresses_by_origin.items()
        }

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.NetworkStream:
        candidates = self._addresses_by_origin.get((_canonical_host(host), int(port)))
        if not candidates:
            raise httpcore.ConnectError("provider address was not pinned")

        last_error: OSError | None = None
        for candidate in candidates:
            sock = socket.socket(candidate.family, socket.SOCK_STREAM)
            try:
                if socket_options:
                    for option in socket_options:
                        sock.setsockopt(*option)
                if timeout is not None:
                    sock.settimeout(timeout)
                if local_address:
                    if candidate.family == socket.AF_INET6:
                        sock.bind((local_address, 0, 0, 0))
                    else:
                        sock.bind((local_address, 0))
                sock.connect(candidate.sockaddr)
                return SyncStream(sock)
            except OSError as exc:
                last_error = exc
                sock.close()
        raise httpcore.ConnectError("provider connection failed") from last_error


class PinnedHTTPTransport(httpx.BaseTransport):
    """httpx transport with a fixed DNS result and no proxy support."""

    def __init__(self, *, host: str, port: int, addresses: Iterable[ResolvedAddress]):
        self._transport = httpx.HTTPTransport(trust_env=False, proxy=None)
        # HTTPTransport intentionally keeps the pool private. httpx 0.28 has no
        # public network-backend injection point, so replace the empty pool's
        # backend before the first request. The pool remains the stock, tested
        # implementation for TLS, Host, SNI, and response handling.
        self._transport._pool._network_backend = PinnedNetworkBackend(  # type: ignore[attr-defined]
            {(host, int(port)): tuple(addresses)}
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._transport.handle_request(request)

    def close(self) -> None:
        self._transport.close()


def build_pinned_transport(url: str) -> PinnedHTTPTransport:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise UnsafeProviderAddress("provider verification requires HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise UnsafeProviderAddress(
            "provider URL must not contain credentials or query data"
        )
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise UnsafeProviderAddress("provider URL has an invalid port") from exc
    addresses = resolve_public_addresses(parsed.hostname, port)
    return PinnedHTTPTransport(host=parsed.hostname, port=port, addresses=addresses)


def build_explicit_proxy_transport(proxy_url: str) -> httpx.HTTPTransport:
    """Build a proxy transport only when the operator explicitly opts in.

    The normal verification path remains IP-pinned. This narrow escape hatch is
    for development environments whose DNS is supplied by a local proxy; callers
    must decide which provider adapters are allowed to use it.
    """

    parsed = urlsplit(str(proxy_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeProviderAddress("provider verification proxy must be an HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise UnsafeProviderAddress(
            "provider verification proxy must not contain credentials or query data"
        )
    try:
        parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeProviderAddress("provider verification proxy has an invalid port") from exc
    return httpx.HTTPTransport(proxy=parsed.geturl(), trust_env=False)
