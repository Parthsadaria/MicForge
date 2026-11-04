"""Thread-safe float32 ring buffer used to hand audio from the input callback to output callbacks."""
from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self._cap = int(capacity)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._r = 0
        self._w = 0
        self._n = 0
        self._lock = threading.Lock()

    @property
    def fill(self) -> int:
        return self._n

    def clear(self) -> None:
        with self._lock:
            self._r = self._w = self._n = 0

    def write(self, x: np.ndarray) -> int:
        """Append samples. Returns how many *old* samples were dropped to make room."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] > self._cap:
            x = x[-self._cap :]
        n = x.shape[0]
        with self._lock:
            dropped = max(0, self._n + n - self._cap)
            if dropped:
                self._r = (self._r + dropped) % self._cap
                self._n -= dropped
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
        with self._lock:
            got = min(n, self._n)
            first = min(got, self._cap - self._r)
            out[:first] = self._buf[self._r : self._r + first]
            if first < got:
                out[first:got] = self._buf[: got - first]
            self._r = (self._r + got) % self._cap
            self._n -= got
        return out, got
