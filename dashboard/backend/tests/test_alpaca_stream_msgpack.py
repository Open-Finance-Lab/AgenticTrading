"""alpaca-py's live data stream still decodes its msgpack frames.

``msgpack`` is in ``requirements.txt`` only because alpaca-py's websocket
decoder (``alpaca/data/live/websocket.py``) needs it. Nothing in the backend
imports it or a ``*DataStream``, so no other test loads it. The pin moves for
C-extension security fixes (GHSA-j586-36cw-2gc2 in 1.2.3, GHSA-j8c6-9vc5-hj6x
in 1.2.4), and those releases also changed the C decoder this path runs:
nanosecond validation on Timestamp unpacking (1.2.2), new ``ValueError``s on
Unpacker misuse (1.2.3), and capped container preallocation (1.2.4). A green
suite therefore said nothing about whether a bump kept the stream working.

This drives one frame, shaped like Alpaca's v2 stock stream (an array of
messages, with ``t`` as a msgpack Timestamp ext), through alpaca-py's own
receive loop with a fake socket. No network and no credentials are involved.
"""

import asyncio
from datetime import datetime, timezone

import msgpack
from alpaca.data.live.stock import StockDataStream

BAR_TIME = datetime(2026, 10, 29, 13, 30, tzinfo=timezone.utc)


class _OneFrameSocket:
    """Hands ``_consume`` one frame. A second ``recv`` fails the test."""

    def __init__(self, frame):
        self._frames = [frame]
        self.closed = False

    async def recv(self):
        if not self._frames:
            raise AssertionError("stream read past the only frame")
        return self._frames.pop()

    async def close(self):
        self.closed = True


def test_receive_loop_decodes_a_bar_frame():
    stream = StockDataStream("test-key", "test-secret")
    received = []

    async def on_bar(bar):
        received.append(bar)
        # Asks _consume to close the socket before its next recv.
        stream._stop_stream_queue.put_nowait({"should_stop": True})

    stream.subscribe_bars(on_bar, "SPY")
    frame = msgpack.packb(
        [
            {"T": "subscription", "bars": ["SPY"]},
            {
                "T": "b",
                "S": "SPY",
                "o": 580.1,
                "h": 580.9,
                "l": 579.8,
                "c": 580.5,
                "v": 12345,
                "t": msgpack.Timestamp.from_datetime(BAR_TIME),
                "n": 321,
                "vw": 580.4,
            },
        ]
    )
    socket = _OneFrameSocket(frame)
    stream._ws = socket

    asyncio.run(stream._consume())

    assert socket.closed
    assert len(received) == 1
    bar = received[0]
    assert bar.symbol == "SPY"
    assert bar.timestamp == BAR_TIME
    assert (bar.open, bar.high, bar.low, bar.close) == (580.1, 580.9, 579.8, 580.5)
    assert (bar.volume, bar.trade_count, bar.vwap) == (12345, 321, 580.4)
