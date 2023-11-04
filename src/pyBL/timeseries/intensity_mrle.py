from __future__ import annotations

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt

from pyBL.raincell.cell import Cell, ConstantCell, CType

__all__ = ["IntensityMRLE"]

CDelta = Callable[[CType], List[Tuple[float, float]]]

IMRLESequence = Optional[Union[npt.NDArray[np.float64], List[float]]]


class IntensityMRLE:
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

    __slots__ = ("_time", "_intensity", "_intensity_delta", "_scale")
    _cell_register = {}  # type: ignore

    def __init__(
        self,
        time: IMRLESequence = None,
        intensity: IMRLESequence = None,
        scale: float = 1
    ):
        """
        Parameters
        ----------
        time: Optional[Union[`np.ndarray`, `List`]]
            Time index of the intensity timeseries.

        intensity: Optional[Union[`np.ndarray`, `List`]]
            Intensity values (mm/h) of the intensity timeseries.
        """

        self._time, self._intensity = _mrle_check(time, intensity)
        self._intensity_delta: npt.NDArray[np.float64] = np.column_stack(
            (self._time[: -1], np.diff(self._intensity[: -1], prepend=0))
        )
        self._scale = scale

    @classmethod
    def fromDelta(
        cls,
        time: IMRLESequence = None,
        intensity_delta: IMRLESequence = None,
        scale: int = 1
    ) -> IntensityMRLE:
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
        time, intensity_delta = _delta_to_mrle(time, intensity_delta)

        return cls(time, intensity_delta, scale)

    @classmethod
    def fromCells(cls, cells: List[CType], scale: int = 1) -> IntensityMRLE:
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
            scale=scale
        )

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
        return self._time

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        return self._intensity

    @property
    def intensity_delta(self) -> npt.NDArray[np.float64]:
        return self._intensity_delta

    def add(self, cell: Cell) -> None:
        try:
            delta_func = self._cell_register[type(cell)]
        except KeyError:
            raise KeyError(
                f"Cell type {type(cell)} is not registered with IntensityDelta"
            )

        delta_encoding = np.array(delta_func(cell), dtype=np.float64)

        self._intensity_delta = np.concatenate((self._intensity_delta, delta_encoding))

        _time, _intensity = _delta_to_mrle(
            time=self._intensity_delta[:, 0],
            intensity_delta=self._intensity_delta[:, 1],
        )

        self._time, self._intensity = _mrle_check(_time, _intensity)

    def __add__(self, cell: Cell) -> IntensityMRLE:
        try:
            delta_func = self._cell_register[type(cell)]
        except KeyError:
            raise KeyError(
                f"Cell type {type(cell)} is not registered with IntensityDelta"
            )

        delta_encoding = np.array(delta_func(cell), dtype=np.float64)
        _intensity_delta = np.concatenate((self._intensity_delta, delta_encoding))

        _time, _intensity = _delta_to_mrle(
            time=_intensity_delta[:, 0], intensity_delta=_intensity_delta[:, 1]
        )

        return type(self)(_time, _intensity, self._scale)

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
            if idx == 0 or idx == len(self._time):
                return 0.0
            return self._intensity[idx - 1]
        else:
            raise TypeError("time_idx must be an int or a slice")

    def __str__(self) -> str:
        time = "Time: " + " ".join(f"{num:>{5}}" for num in self.time)
        intensity = "Intensity: " + " ".join(f"{num:>{5}}" for num in self.intensity)

        return time + "\n" + intensity + "\n"
    
    def mean(self) -> float:
        return _mrle_mean(self._time, self._intensity) / self._scale
    
    def acf(self, lag=1) -> float:
        return _mrle_acf(self._time, self._intensity, lag=lag)
    
    def cvar(self, ddof=1) -> float:
        return _mrle_cvar(self._time, self._intensity, ddof=ddof)
    
    def skew(self) -> float:
        return _mrle_skew(self._time, self._intensity)
    
    def pDry(self) -> float:
        return _mrle_pDry(self._time, self._intensity)
    
    def rescale(self, scale: float) -> IntensityMRLE:
        scale_time, scale_intensity = _mrle_rescale(self._time, self._intensity, scale)
        return type(self)(scale_time, scale_intensity, self._scale * scale)

    

def _mrle_check(
    time: IMRLESequence = None,
    intensity: IMRLESequence = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Check if both time and intensity are None
    if time is None or intensity is None:
        if time is not None or intensity is not None:
            raise ValueError("time and intensity must both be None or not None")

        _time: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

        _intensity: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

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
    # Add ending time and intensity if there isn't one
    if not np.isnan(intensity[-1]):
        time = np.append(time, time[-1] + 1)
        intensity = np.append(intensity, np.nan)

    # Make sure it's in MRLE format
    intensity_idx = np.diff(intensity, prepend=-np.inf) != 0
    time = time[intensity_idx]
    intensity = intensity[intensity_idx]

    #_time = np.insert(time, [0, time.size], [-np.inf, np.inf])
    #_intensity = np.insert(intensity, [0, intensity.size], [0, 0])

    return time, intensity


@overload
def _delta_to_mrle(
    time: IMRLESequence,
    intensity_delta: IMRLESequence,
):
    ...


@overload
def _delta_to_mrle(
    time: None, intensity_delta: None, delta_encoding: npt.NDArray[np.float64]
):
    ...


def _delta_to_mrle(
    time: IMRLESequence = None,
    intensity_delta: IMRLESequence = None,
    delta_encoding: Optional[npt.NDArray[np.float64]] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not isinstance(time, np.ndarray):
        time = np.array(time, dtype=np.float64)
    if not isinstance(intensity_delta, np.ndarray):
        intensity_delta = np.array(intensity_delta, dtype=np.float64)

    # Zip time and intensity_delta into a 2D array
    delta_encoding = np.column_stack((time, intensity_delta))

    # Sort the change_time_idx by time
    delta_encoding = delta_encoding[np.argsort(delta_encoding[:, 0])]

    # Calculate the cumulative sum of the intensity changes inplace.
    np.cumsum(delta_encoding[:, 1], out=delta_encoding[:, 1])

    # TODO: Round to 10 decimal places to avoid floating point errors. And it's inplace.
    delta_encoding.round(10, out=delta_encoding)

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

@IntensityMRLE.register_cell(ConstantCell)
def constant_cell_delta(cell: ConstantCell) -> List[tuple[float, float]]:
    return [(cell.start, cell.intensity), (cell.end, -cell.intensity)]

def _mrle_mean(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]) -> float:
    each_intensity_duration = np.diff(time)
    return np.sum(intensity[:-1] * each_intensity_duration) / (time[-1] - time[0])

def _mrle_acf(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], lag=1, mean = None, sse = None):
    if lag == 0:
        return 1
    n = len(time)

    if mean is None:
        sum = 0
        for i in range(n-1):
            sum += intensity[i] * (time[i+1] - time[i])
        mean = sum / (time[-1] - time[0]) #Mean
        
    if sse is None:
        sse = 0
        for i in range(n-1):
            sse += (intensity[i] - mean)**2 * (time[i+1] - time[i])
    
    shift = 0
    for time_idx in time:
        if time_idx < lag + time[0]:
            shift += 1
        else:
            break
    
    x_idx = shift
    y_idx = 1
    result = 0
    while x_idx < n:
        if time[y_idx] >= time[x_idx] - lag:
            size = (time[x_idx] - lag) - max(time[y_idx-1], (time[x_idx-1]-lag))
            result += ((intensity[y_idx-1]-mean) * (intensity[x_idx-1]-mean)) * size
            x_idx += 1
        else:
            size = time[y_idx] - max(time[y_idx-1], (time[x_idx-1]-lag))
            result += ((intensity[y_idx-1]-mean) * (intensity[x_idx-1]-mean)) * size
            y_idx += 1
    return result / sse

def _mrle_cvar(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None, sse = None, ddof = 0) -> float:
    # Coefficient of variation
    if mean is None:
        sum = 0
        for i in range(len(time)-1):
            sum += intensity[i] * (time[i+1] - time[i])
        mean = sum / (time[-1] - time[0]) #Mean
        
    if sse is None:
        sse = 0
        for i in range(len(time)-1):
            sse += (intensity[i] - mean)**2 * (time[i+1] - time[i])

    return (sse / (time[-1] - time[0] - ddof))**0.5 / mean

def _mrle_skew(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None, sse = None) -> float:
    # Skewness
    n = time[-1] - time[0]
    if mean is None:
        sum = 0
        for i in range(len(time)-1):
            sum += intensity[i] * (time[i+1] - time[i])
        mean = sum / n

    # Sum of squared error
    if sse is None:
        sse = 0
        for i in range(len(time)-1):
            sse += (intensity[i] - mean)**2 * (time[i+1] - time[i])

    # Standard deviation
    # TODO: Check how to do unbiased standard deviation on MRLE when time is float
    sd = (sse / (n - 1))**0.5 

    # Skewness
    skewness = 0
    for i in range(len(time)-1):
        skewness += (intensity[i] - mean)**3 * (time[i+1] - time[i])
    skewness = skewness/(((time[-1] - time[0]))*sd**3)
    return skewness

def _mrle_pDry(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]) -> float:
    wet_time = 0
    for i in range(len(time)-1):
        if intensity[i] > 0:
            wet_time += time[i+1] - time[i]

    return 1 - (wet_time / (time[-1] - time[0]))

def _mrle_rescale(
    time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], scale: float
):
    n = len(time) - 1
    scale_time = np.zeros(n*3+2)
    scale_intensity = np.zeros(n*3+2)

    scale_time[0] = -1
    rescale_idx = 1
    for i in range(n):
        srt, end = time[i], time[i+1]
        r_srt, r_end = srt//scale, (end - 1)//scale
        intensity_i = intensity[i]
        if r_srt == r_end:
            if scale_time[rescale_idx - 1] == r_srt:
                rescale_idx -= 1
            scale_time[rescale_idx] = r_srt
            scale_intensity[rescale_idx] += intensity_i*(end-srt)
            rescale_idx += 1
        
        if r_srt + 1 == r_end:
            if scale_time[rescale_idx - 1] == r_srt:
                rescale_idx -= 1
            scale_time[rescale_idx] = r_srt
            scale_intensity[rescale_idx] += intensity_i*(scale - srt % scale)
            rescale_idx += 1
            scale_time[rescale_idx] = r_end
            scale_intensity[rescale_idx] += intensity_i*((end-1) % scale + 1)
            rescale_idx += 1
            
        if r_srt + 1 < r_end:
            if scale_time[rescale_idx - 1] == r_srt:
                rescale_idx -= 1
            scale_time[rescale_idx] = r_srt
            scale_intensity[rescale_idx] += intensity_i*(scale - srt % scale)
            rescale_idx += 1
            scale_time[rescale_idx] = r_srt + 1
            scale_intensity[rescale_idx] += intensity_i*scale
            rescale_idx += 1
            scale_time[rescale_idx] = r_end
            scale_intensity[rescale_idx] += intensity_i*((end-1) % scale + 1)
            rescale_idx += 1

    scale_time[rescale_idx] = scale_time[rescale_idx - 1] + 1
    scale_intensity[rescale_idx] = np.nan
    scale_time = scale_time[1:rescale_idx+1]
    scale_intensity = scale_intensity[1:rescale_idx+1]

    # Preallocate array
    diff_time = np.array([True] * (len(scale_time)))
    for i in range(1, len(scale_time)):
        diff_time[i] = (scale_intensity[i] != scale_intensity[i-1])
    diff_time[0] = True

    scale_time = scale_time[diff_time]
    scale_intensity = scale_intensity[diff_time]

    return scale_time , scale_intensity