"""Thread-safe float32 ring buffer used to hand audio from the input callback to output callbacks.

Lock-free Single-Producer Single-Consumer (SPSC) design:
The write side (input callback) owns `_w`, and the read side (output callback) owns `_r`.
Both pointers increase monotonically without wrap-around (Python arbitrary-precision integers
never overflow 32-bit or 64-bit bounds).  Fill level is computed directly as `_w - _r`,
completely eliminating shared mutable counters (`_n`) and cross-thread GIL race conditions
which previously caused false buffer starvation and runaway underrun cascades.
"""
from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._r = 0
        self._w = 0

    @property
    def fill(self) -> int:
        return max(0, self._w - self._r)

    def clear(self) -> None:
        self._r = self._w

    def write(self, x: np.ndarray) -> int:
        """Append samples. Returns how many *old* samples were dropped to make room."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] > self._cap:
            x = x[-self._cap :]
        n = x.shape[0]
        w = self._w
        r = self._r

        # Overflow: drop oldest samples to make room
        dropped = max(0, (w + n) - r - self._cap)
        if dropped:
            self._r = r + dropped

        # Copy data into the circular buffer
        w_idx = w % self._cap
        first = min(n, self._cap - w_idx)
        self._buf[w_idx : w_idx + first] = x[:first]
        if first < n:
            self._buf[: n - first] = x[first:]
        self._w = w + n
        return dropped

    def read(self, n: int) -> tuple[np.ndarray, int]:
        """Read up to n samples. Returns (array of length n, zero-padded; samples actually read)."""
        out = np.zeros(n, dtype=np.float32)
        w = self._w
        r = self._r
        avail = max(0, w - r)
        got = min(n, avail)
        if got > 0:
            r_idx = r % self._cap
            first = min(got, self._cap - r_idx)
            out[:first] = self._buf[r_idx : r_idx + first]
            if first < got:
                out[first:got] = self._buf[: got - first]
            self._r = r + got
        return out, got

    def read_into(self, dest: np.ndarray) -> int:
        """Read up to len(dest) samples directly into a pre-allocated buffer.

        This avoids allocating a new array on every output callback – critical
        for real-time audio threads where allocation → GC → click.
        Returns the number of samples actually read; unread portion of dest is zeroed.
        """
        n = dest.shape[0]
        w = self._w
        r = self._r
        avail = max(0, w - r)
        got = min(n, avail)
        if got > 0:
            r_idx = r % self._cap
            first = min(got, self._cap - r_idx)
            dest[:first] = self._buf[r_idx : r_idx + first]
            if first < got:
                dest[first:got] = self._buf[: got - first]
            self._r = r + got
        if got < n:
            dest[got:] = 0.0
        return got
