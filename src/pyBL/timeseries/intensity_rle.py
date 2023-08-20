from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional, Union, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..dataclasses.storm import Cell, Storm


class IntensityRLE:
    """
    Represent a intensity timeseries as a *modified* Run-Length Encoded array.

    Normal RLE stores the length of a run of repeated values.

    In this class, we store the index of the start of a run of repeated values.

    For example, if we have a intensity timeseries of
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    then the RLE representation is
        [(3, 0), (2, 1), (5, 0)]

    and the IntensityRLE representation is
        [(0, 0), (3, 1), (5, 0)]


    """

    __slots__ = ("_time", "_intensity", "_scale")

    def __init__(
        self,
        time: Optional[Union[npt.NDArray[np.float64], list[float]]] = None,
        intensity: Optional[Union[npt.NDArray[np.float64], list[float]]] = None,
        timescale: timedelta = timedelta(hours=1),
    ):
        """
        Parameters
        ----------
        time: Optional[Union[`np.ndarray`, `list`]]
            Time index of the intensity timeseries.

        intensity: Optional[Union[`np.ndarray`, `list`]]
            Intensity values (mm/h) of the intensity timeseries.

        timescale: `timedelta`
            The timescale of the time index.
            Default: `timedelta(hours=1)`
        """

        # Check if both time and intensity are None
        if time is None or intensity is None:
            if time is not None or intensity is not None:
                raise ValueError("time and intensity must both be None or not None")
            self._time: npt.NDArray[np.float64] = np.array(
                [-np.inf, np.inf], dtype=np.float64
            )
            self._intensity: npt.NDArray[np.float64] = np.array(
                [0, 0], dtype=np.float64
            )
            return

        # Check if time and intensity have the same length
        if len(time) != len(intensity):
            raise ValueError("time and intensity must have the same length")

        # Check if the RLE time is monotonically increasing.
        if np.any(np.diff(time) <= 0):
            raise ValueError("time must be monotonically increasing")

        # Turn time and intensity into numpy arrays
        self._time = np.array(time, dtype=np.float64)
        self._intensity = np.array(intensity, dtype=np.float64)

        # Check if time and intensity are 1D arrays
        if self._time.ndim != 1 or self._intensity.ndim != 1:
            raise ValueError("time and intensity must be 1D arrays")

        # Check if _time have leading and trailing infinities.
        # We've already checked that time and intensity have the same length,
        # so if _time didn't have leading and trailing infinities,
        # then we can just insert them both into _time and _intensity.
        if self._time[0] != -np.inf:
            self._time = np.insert(self._time, 0, -np.inf)
            self._intensity = np.insert(self._intensity, 0, 0)
        if self._time[-1] != np.inf:
            self._time = np.append(self._time, np.inf)
            self._intensity = np.append(self._intensity, 0)

    @classmethod
    def fromCells(
        cls, cells: list[Cell], timescale: Optional[timedelta] = timedelta(hours=1)
    ) -> IntensityRLE:
        """

        Parameters:
        ----------
            cells: list[`Cell`]:
                List of cells to convert to an IntensityRLE.

            timescale: Optional[`timedelta`]
                The timescale of the time index.
                If None, the timescale of the first cell is used.
                Defaults to None.
        """
        # Preallocate the time that the intensity changes.
        # Size is 2 * len(cells). Shape is (len(cells), 2).
        change_time_idx = np.empty((len(cells) * 2, 2), dtype=np.float64)

        if timescale is None:
            timescale = cells[0].timescale
        for idx, cell in enumerate(cells):
            change_time_idx[idx] = [cell.startTime, cell.intensity]
            change_time_idx[idx + len(cells)] = [cell.endTime, -cell.intensity]

        # Sort the change_time_idx by time
        sorted_idx = np.argsort(change_time_idx[:, 0])
        change_time_idx = change_time_idx[sorted_idx]

        # Calculate the cumulative sum of the intensity changes
        change_time_idx[:, 1] = np.cumsum(change_time_idx[:, 1])

        # Remove duplicate times by keeping the last occurence
        # Since the np.unique function only keeps the first occurence,
        # We reverse the array, keep the first occurence, then reverse it back.
        _, unique_idx = np.unique(change_time_idx[:, 0][::-1], return_index=True)
        # Adjust indices to account for reversed order
        unique_indices = len(change_time_idx) - 1 - unique_idx

        return cls(
            time=change_time_idx[unique_indices, 0],
            intensity=change_time_idx[unique_indices, 1],
            timescale=timescale,
        )

    @classmethod
    def fromStorms(cls, storms: list[Storm]) -> IntensityRLE:
        raise NotImplementedError

    def _insert_time(
        self, array: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        for value in values:
            idx = np.searchsorted(array, value)

            if idx < len(array) and array[idx] == value:
                continue

            array = np.insert(array, idx, value)
        return array

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return self._time[1:-1]

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        return self._intensity[1:-1]

    def add(self, cell: Cell) -> None:
        if cell.startTime >= cell.endTime:
            raise ValueError("Cell start time must be less than end time")
        insert_sDT = np.searchsorted(self._time, cell.startTime)
        appear_sDT = self._time[insert_sDT] == cell.startTime
        insert_eDT = np.searchsorted(self._time, cell.endTime)
        appear_eDT = self._time[insert_eDT] == cell.endTime

        insert_intensity = (
            self._intensity[insert_sDT - (0 if appear_sDT else 1) : insert_eDT]
            + cell.intensity
        )

        self._time = self._insert_time(
            self._time, np.array([cell.startTime, cell.endTime])
        )
        self._intensity = np.concatenate(
            (
                self._intensity[:insert_sDT],
                insert_intensity,
                self._intensity[insert_eDT - (0 if appear_eDT else 1) :],
            )
        )

    @overload
    def __getitem__(self, time_idx: slice) -> npt.NDArray[np.float64]:
        ...

    @overload
    def __getitem__(self, time_idx: int) -> np.float64:
        ...

    def __getitem__(self, time_idx: Any) -> Any:
        if isinstance(time_idx, slice):
            """
            TODO: Implement this
            """
            raise NotImplementedError
        elif isinstance(time_idx, int):
            # Use binary search to find the index
            idx = np.searchsorted(a=self._time, v=time_idx, side="right")
            return self._intensity[idx - 1]
        else:
            raise TypeError("time_idx must be an int or a slice")

    def __add__(self, cell: Cell) -> IntensityRLE:
        if cell.startTime >= cell.endTime:
            raise ValueError("Cell start time must be less than end time")

        insert_start = np.searchsorted(self._time, cell.startTime)
        appear_start = self._time[insert_start] == cell.startTime
        insert_end = np.searchsorted(self._time, cell.endTime)
        appear_end = self._time[insert_end] == cell.endTime

        insert_intensity = (
            self._intensity[insert_start - (0 if appear_start else 1) : insert_end]
            + cell.intensity
        )
        # Insert cell.sDT and cell.eDT into a sorted array self.time_idx

        return IntensityRLE(
            self._insert_time(self._time, np.array([cell.startTime, cell.endTime])),
            np.concatenate(
                (
                    self._intensity[:insert_start],
                    insert_intensity,
                    self._intensity[insert_end - (0 if appear_end else 1) :],
                )
            ),
        )

    def __repr__(self) -> str:
        time = "Time: " + " ".join(f"{num:>{5}}" for num in self.time)
        intensity = "Intensity: " + " ".join(f"{num:>{5}}" for num in self.intensity)

        return time + "\n" + intensity + "\n"
