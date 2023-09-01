from __future__ import annotations

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt

from ..raincell.cell import Cell, ConstantCell

CType = TypeVar("CType", bound=Cell[Any])
CDelta = Callable[[CType], List[Tuple[float, float]]]


class IntensityDelta:
    """
    This class stores timeseries in Modified Run-Length Encoding.

    Normal RLE stores the length of a run of repeated values.

    In this class, we store the index of the start of a run of repeated values.

    For example, if we have a intensity timeseries of
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    then the RLE representation is
        [(3, 0), (2, 1), (5, 0)]

    and the IntensityRLE representation is
        [(0, 0), (3, 1), (5, 0)]


    """

    __slots__ = ("_time", "_intensity", "_intensity_delta")
    _cell_register = {}  # type: ignore

    def __init__(
        self,
        time: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
        intensity: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
    ):
        """
        Parameters
        ----------
        time: Optional[Union[`np.ndarray`, `List`]]
            Time index of the intensity timeseries.

        intensity: Optional[Union[`np.ndarray`, `List`]]
            Intensity values (mm/h) of the intensity timeseries.
        """

        self._time, self._intensity = self._mrle_check(time, intensity)
        self._intensity_delta: npt.NDArray[np.float64] = np.column_stack(
            (self._time[1:-1], np.diff(self._intensity[1:-1], prepend=0))
        )

    @classmethod
    def fromDelta(
        cls,
        time: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
        intensity_delta: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
    ) -> IntensityDelta:
        """
        Parameters:
        ----------
        time: Optional[Union[`np.ndarray`, `List`]]
            Time index of the intensity timeseries.
            The order of the times does not matter.
            Duplicate times will be summed together.

        intensity_delta: Optional[Union[`np.ndarray`, `List`]]
            Delta Encoded Intensity values (mm/h) of the intensity timeseries.

        """
        time, intensity_delta = cls._delta_to_mrle(time, intensity_delta)

        return cls(time, intensity_delta)

    @staticmethod
    def _mrle_check(
        time: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
        intensity: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Check if both time and intensity are None
        if time is None or intensity is None:
            if time is not None or intensity is not None:
                raise ValueError("time and intensity must both be None or not None")

            _time: npt.NDArray[np.float64] = np.array(
                [-np.inf, np.inf], dtype=np.float64
            )

            _intensity: npt.NDArray[np.float64] = np.array([0, 0], dtype=np.float64)

            return _time, _intensity
        else:
            time = np.array(time, dtype=np.float64)
            intensity = np.array(intensity, dtype=np.float64)

        # Check if time and intensity have the same length
        if time.size != intensity.size:
            raise ValueError("time and intensity must have the same length")
        # Check if time and intensity are 1D arrays
        if time.ndim != 1 or intensity.ndim != 1:
            raise ValueError("time and intensity must be 1D arrays")
        # Check if the RLE time is strictly increasing.
        if np.any(np.diff(time) < 0):
            raise ValueError("time must be strictly increasing")
        # Check if last intensity is 0
        if intensity[-1] != 0:
            raise ValueError("Last intensity must be 0")

        _time = np.insert(time, [0, time.size], [-np.inf, np.inf])
        _intensity = np.insert(intensity, [0, intensity.size], [0, 0])

        return _time, _intensity

    @staticmethod
    def _delta_to_mrle(
        time: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
        intensity_delta: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        time = np.array(time, dtype=np.float64)
        intensity_delta = np.array(intensity_delta, dtype=np.float64)

        # Zip time and intensity_delta into a 2D array
        delta_encoding = np.column_stack((time, intensity_delta))

        # Sort the change_time_idx by time
        sorted_idx = np.argsort(delta_encoding[:, 0])
        delta_encoding = delta_encoding[sorted_idx]

        # Calculate the cumulative sum of the intensity changes
        delta_encoding[:, 1] = np.cumsum(delta_encoding[:, 1])

        # Remove duplicate times by keeping the last occurence
        # Since the np.unique function only keeps the first occurence,
        # We reverse the array, keep the first occurence, then reverse it back.
        _, unique_idx = np.unique(delta_encoding[:, 0][::-1], return_index=True)
        # Adjust indices to account for reversed order
        unique_indices = len(delta_encoding) - 1 - unique_idx
        return (
            delta_encoding[:, 0][unique_indices],
            delta_encoding[:, 1][unique_indices],
        )

    @classmethod
    def fromCells(cls, cells: List[CType]) -> IntensityDelta:
        """
        Parameters:
        ----------
            cells: List[`Cell`]:
                List of cells to convert to an IntensityRLE.
        """
        # Get the register function for the cell type
        try:
            delta_func = cls._cell_register[type(cells[0])]
        except KeyError:
            raise KeyError(
                f"Cell type {type(cells[0])} is not registered with IntensityDelta"
            )

        delta_encoding = [
            [time, intensity_delta]
            for cell in cells
            for time, intensity_delta in delta_func(cell)
        ]

        delta_encoding_np: npt.NDArray[np.float64] = np.array(
            delta_encoding, dtype=np.float64
        )

        return cls.fromDelta(
            time=delta_encoding_np[:, 0],
            intensity_delta=delta_encoding_np[:, 1],
        )

    def _insert_time(
        self, array: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        for value in values:
            idx = np.searchsorted(array, value)

            if idx < len(array) and array[idx] == value:
                continue

            array = np.insert(array, idx, value)
        return array

    @classmethod
    def register_cell(
        cls, cell_type: Type[CType]
    ) -> Callable[[CDelta[CType]], CDelta[CType]]:
        def decorator(func: CDelta[CType]) -> CDelta[CType]:
            cls._cell_register[cell_type] = func
            return func

        return decorator

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return self._time[1:-1]

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        return self._intensity[1:-1]

    @property
    def intensity_delta(self) -> npt.NDArray[np.float64]:
        return self._intensity_delta

    def add(self, cell: Cell[Any]) -> None:
        try:
            delta_func = self._cell_register[type(cell)]
        except KeyError:
            raise KeyError(
                f"Cell type {type(cell)} is not registered with IntensityDelta"
            )

        delta_encoding = np.array(delta_func(cell), dtype=np.float64)

        self._intensity_delta = np.concatenate((self._intensity_delta, delta_encoding))

        _time, _intensity = self._delta_to_mrle(
            time=self._intensity_delta[:, 0],
            intensity_delta=self._intensity_delta[:, 1],
        )
        self._time, self._intensity = self._mrle_check(_time, _intensity)

    def __add__(self, cell: Cell[Any]) -> IntensityDelta:
        try:
            delta_func = self._cell_register[type(cell)]
        except KeyError:
            raise KeyError(
                f"Cell type {type(cell)} is not registered with IntensityDelta"
            )

        delta_encoding = np.array(delta_func(cell), dtype=np.float64)

        _intensity_delta = np.concatenate((self._intensity_delta, delta_encoding))

        _time, _intensity = self._delta_to_mrle(
            time=_intensity_delta[:, 0], intensity_delta=_intensity_delta[:, 1]
        )

        return type(self)(_time, _intensity)

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

    def __str__(self) -> str:
        time = "Time: " + " ".join(f"{num:>{5}}" for num in self.time)
        intensity = "Intensity: " + " ".join(f"{num:>{5}}" for num in self.intensity)

        return time + "\n" + intensity + "\n"


@IntensityDelta.register_cell(ConstantCell)
def constant_cell_delta(cell: ConstantCell) -> List[tuple[float, float]]:
    return [(cell.start, cell.intensity_args), (cell.end, -cell.intensity_args)]
