"""Thread-safe float32 ring buffer used to hand audio from the input callback to output callbacks.

Lock-free design: the write side (input callback) and read side (output callback)
operate on independent atomic pointers.  The only shared variable is `_n` (fill level),
which is updated with a single integer store – safe on CPython due to the GIL guarantee
that int loads/stores are atomic.  This avoids lock contention between PortAudio threads
which was a major source of output underruns and audible clicks.
"""
from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._r = 0
        self._w = 0
        self._n = 0  # atomic on CPython (int store/load under GIL)

    @property
    def fill(self) -> int:
        return self._n

    def clear(self) -> None:
        self._r = 0
        self._w = 0
        self._n = 0

    def write(self, x: np.ndarray) -> int:
        """Append samples. Returns how many *old* samples were dropped to make room."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] > self._cap:
            x = x[-self._cap :]
        n = x.shape[0]

        # Overflow: drop oldest samples to make room
        dropped = max(0, self._n + n - self._cap)
        if dropped:
            self._r = (self._r + dropped) % self._cap
            self._n -= dropped

        # Copy data into the circular buffer
        first = min(n, self._cap - self._w)
        self._buf[self._w : self._w + first] = x[:first]
        if first < n:
            self._buf[: n - first] = x[first:]
        self._w = (self._w + n) % self._cap
        self._n += n
        return dropped

    def read(self, n: int) -> tuple[np.ndarray, int]:
        """Read up to n samples. Returns (array of length n, zero-padded; samples actually read)."""
        out = np.zeros(n, dtype=np.float32)
        got = min(n, self._n)
        if got > 0:
            first = min(got, self._cap - self._r)
            out[:first] = self._buf[self._r : self._r + first]
            if first < got:
                out[first:got] = self._buf[: got - first]
            self._r = (self._r + got) % self._cap
            self._n -= got
        return out, got

    def read_into(self, dest: np.ndarray) -> int:
        """Read up to len(dest) samples directly into a pre-allocated buffer.

        This avoids allocating a new array on every output callback – critical
        for real-time audio threads where allocation → GC → click.
        Returns the number of samples actually read; unread portion of dest is zeroed.
        """
        n = dest.shape[0]
        got = min(n, self._n)
        if got > 0:
            first = min(got, self._cap - self._r)
            dest[:first] = self._buf[self._r : self._r + first]
            if first < got:
                dest[first:got] = self._buf[: got - first]
            self._r = (self._r + got) % self._cap
            self._n -= got
        if got < n:
            dest[got:] = 0.0
        return got
