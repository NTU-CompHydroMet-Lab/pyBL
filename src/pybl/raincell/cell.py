from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypeVar

# Intensity arguments type
CType = TypeVar("CType", bound="Cell")
CType_co = TypeVar("CType_co", bound="Cell", covariant=True)


@dataclass
class Cell:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Cell start time must be less than end time")


@dataclass
class ConstantCell(Cell):
    intensity: float


@dataclass(frozen=True)
class Storm:
    start_time: int
    end_time: int
    cells: List[Cell]

    def __post_init__(self) -> None:
        if self.start_time > self.end_time:
            raise ValueError("Cell start time must be less than end time")
