"""Processor interface: one mono float32 block in, same-length block out."""
from __future__ import annotations

import abc

import numpy as np


class Processor(abc.ABC):
    name: str = "processor"

    def __init__(self, sample_rate: int, enabled: bool = True) -> None:
        self.sample_rate = int(sample_rate)
        self.enabled = enabled

    @property
    def latency_samples(self) -> int:
        """Algorithmic delay introduced by this stage."""
        return 0

    @abc.abstractmethod
    def process(self, x: np.ndarray) -> np.ndarray:
        """Process one block. Must return an array of the same length."""

    def reset(self) -> None:
        """Clear internal state."""

    def telemetry(self) -> dict[str, float]:
        """Live values for the dashboard (gain reduction, etc.)."""
        return {}
