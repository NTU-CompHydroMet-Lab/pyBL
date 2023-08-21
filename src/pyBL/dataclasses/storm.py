from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class Cell:
    startTime: float
    endTime: float
    intensity: float
    timescale: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        if self.startTime > self.endTime:
            raise ValueError("Cell start time must be less than end time")
        if self.intensity <= 0:
            raise ValueError("Cell intensity must be positive")


@dataclass(frozen=True)
class Storm:
    startTime: int
    endTime: int
    cells: list[Cell]
    timescale: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        if self.startTime > self.endTime:
            raise ValueError("Cell start time must be less than end time")
