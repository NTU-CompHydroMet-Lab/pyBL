from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, TypeVar

IntensityType = TypeVar('IntensityType')


@dataclass
class Cell(Generic[IntensityType]):

    start: int
    end: int
    intensity_args: IntensityType

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Cell start time must be less than end time")

@dataclass
class ConstantCell(Cell[float]):
    pass


@dataclass(frozen=True)
class Storm(Generic[IntensityType]):
    start_time: int
    end_time: int
    cells: List[Cell[IntensityType]]

    def __post_init__(self) -> None:
        if self.start_time > self.end_time:
            raise ValueError("Cell start time must be less than end time")
