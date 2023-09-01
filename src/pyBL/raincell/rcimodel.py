from __future__ import annotations

from typing import Protocol


class IRCIModel(Protocol):
    def get_f1(self, x: float) -> float:
        ...
    def get_f2(self, x: float) -> float:
        ...
    def sample_intensity(self):
        ...
