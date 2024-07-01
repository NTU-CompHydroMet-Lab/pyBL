from __future__ import annotations

from datetime import timedelta
from typing import (
    Any,
    List,
    Optional,
    Union,
    overload,
)

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt

__all__ = ["IndexedSnapshot"]

FloatArray = Optional[Union[npt.NDArray[np.float64], List[float]]]


class IndexedSnapshot:
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

    def __init__(
        self,
        time: FloatArray = None,
        intensity: FloatArray = None,
    ):
        """
        Parameters
        ----------
        time: Optional[Union[`np.ndarray`, `List`]]
            Time index of the intensity timeseries.

        intensity: Optional[Union[`np.ndarray`, `List`]]
            Intensity values (mm/h) of the intensity timeseries.
        """
        self._time, self._intensity = _isnapshot_check(time, intensity)
        self._intensity_delta = _intensity_delta_check(self._time, self._intensity)

    @classmethod
    def fromDelta(
        cls, time: FloatArray, intensity_delta: FloatArray
    ) -> IndexedSnapshot:
        """
        Create an IndexedSnapshot timeseries from the delta encoded intensity timeseries.

        Parameters
        ----------
        time: Optional[Union[`np.ndarray`, `List`]]
            Time index of the intensity timeseries.
            The order of the times does not matter.
            Duplicate times will be summed together. (Because it's multiple delta at the same time)

        intensity_delta: Optional[Union[`np.ndarray`, `List`]]
            Delta Encoded Intensity values (mm/h) of the intensity timeseries.

        """
        # Check if time and intensity_delta is np.ndarray
        if not isinstance(time, np.ndarray) or time.dtype != np.float64:
            time = np.array(time, dtype=np.float64)
        if (
            not isinstance(intensity_delta, np.ndarray)
            or intensity_delta.dtype != np.float64
        ):
            intensity_delta = np.array(intensity_delta, dtype=np.float64)

        time, intensity_isnapshot = _delta_to_isnapshot(time, intensity_delta)

        return cls(time, intensity_isnapshot)

    @property
    def time(self) -> npt.NDArray[np.float64]:
        """
        Return the time index of the IndexedSnapshot timeseries.

        Returns
        -------
        time : npt.NDArray[np.float64]
            1D array of time index that follows IndexedSnapshot format.
            Ending time is for indicating the end of the timeseries.
        """
        return self._time

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        """
        Return the intensity of the IndexedSnapshot timeseries.

        Returns
        -------
        intensity : npt.NDArray[np.float64]
            1D array of intensity that follows IndexedSnapshot format.
            Ending intensity is for indicating the end of the timeseries. So it's np.nan.
        """
        return self._intensity

    @property
    def intensity_delta(self) -> npt.NDArray[np.float64]:
        """
        Return the delta encoded intensity of the IndexedSnapshot timeseries.

        Returns
        -------
        intensity_delta : npt.NDArray[np.float64] with shape (n, 2)
            2D array of intensity that follows delta encoding format.
            The first column is the time index.
            The second column is the change in intensity.
        """
        return self._intensity_delta

    def add(self, timeseries: IndexedSnapshot, sequential: bool = False) -> None:
        """
        Add another IndexedSnapshot timeseries to the current IndexedSnapshot timeseries.

        Parameters
        ----------
        timeseries : IndexedSnapshot
            The IndexedSnapshot timeseries to be added.
        sequential : bool, optional
            If False, the timeseries will be added as its original time index.  So overlapping timeseries will be summed together in the overlapping region.
            If True, the timeseries will be added sequentially. So the timeseries will be shifted to the right of the current timeseries.
        """
        if len(timeseries.time) == 0:
            return
        if sequential is True:
            if len(self._time) != 0:
                # Shift the start of the timeseries to be right after th last time of the current timeseries
                timeseries = IndexedSnapshot(
                    timeseries.time - (timeseries.time[0] - self._time[-1]),
                    timeseries.intensity,
                )
        self.__iadd__(timeseries)

    def __iadd__(self, timeseries: IndexedSnapshot) -> IndexedSnapshot:
        result_isnapshot = _merge_isnapshot(self, timeseries)

        self._time, self._intensity = (
            result_isnapshot._time,
            result_isnapshot._intensity,
        )
        self._intensity_delta = result_isnapshot._intensity_delta

        return self

    def __add__(self, timeseries: IndexedSnapshot) -> IndexedSnapshot:
        return _merge_isnapshot(self, timeseries)

    @overload
    def __getitem__(self, time_idx: slice) -> IndexedSnapshot: ...

    @overload
    def __getitem__(self, time_idx: int) -> np.float64: ...

    def __getitem__(self, time_idx: Any) -> Any:
        if isinstance(time_idx, slice):
            return self._get_slice_idx(time_idx)
        elif isinstance(time_idx, int):
            return self._get_int_idx(time_idx)
        else:
            raise TypeError("time_idx must be an int or a slice")

    def _get_slice_idx(self, time_idx: slice) -> IndexedSnapshot:
        """
        ### This is an internal function. You shouldn't use it directly unless you know what you are doing.
        """
        if time_idx.start >= time_idx.stop:
            return type(self)()
        # Use binary search to find the index
        start_idx = np.searchsorted(a=self._time, v=time_idx.start, side="right")
        stop_idx = np.searchsorted(a=self._time, v=time_idx.stop, side="left")

        if start_idx == len(self._time):
            return type(self)()
        if stop_idx == 0:
            return type(self)()

        time = self._time[max(0, start_idx - 1) : stop_idx]
        intensity = self._intensity[max(0, start_idx - 1) : stop_idx]

        if start_idx != 0:
            time[0] = time_idx.start
        if stop_idx < len(self._time):
            time = np.append(time, time_idx.stop)
            intensity = np.append(intensity, np.nan)

        return type(self)(time, intensity)

    def _get_int_idx(self, time_idx: int) -> np.float64:
        """
        ### This is an internal function. You shouldn't use it directly unless you know what you are doing.
        """
        # Use binary search to find the index
        idx = np.searchsorted(a=self._time, v=time_idx, side="right")
        if idx == 0 or idx == len(self._time):
            return np.float64(0.0)
        return self._intensity[int(idx) - 1]

    def __str__(self) -> str:
        time_value = "\n".join(
            f"{self.time[i]} {self.intensity[i]}" for i in range(len(self.time))
        )
        return time_value

    def sum(self) -> float:
        """
        Calculate the total depth of the intensity timeseries.
        This value should roughly be the same after rescaling with any scale.

        Returns
        -------
        sum : float
            The sum of depth of the intensity timeseries.
        """
        if self._time.size == 0:
            return 0
        depth, _ = _isnapshot_sum_and_duration(self._time, self._intensity)
        return depth

    def mean(self) -> float:
        """
        Calculate the mean of the intensity timeseries.

        Returns
        -------
        mean : float
            The mean of the intensity timeseries.
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_mean(self._time, self._intensity)

    def sum_squared_error(self) -> float:
        """
        Calculate the sum of squared error of the intensity timeseries.

        Returns
        -------
        sum of squared error: float
            The sum of squared error of the intensity timeseries.
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_sum_squared_error(self._time, self._intensity)

    def autocorr_coef(self, lag: Optional[float] = 1) -> float:
        """
        Calculate the n-lag autocorrelation coefficient of the intensity timeseries.
        It's basically Cov(X_t, X_{t+lag}) / Var(X_t)

        Parameters
        ----------
        lag : int, optional
            The lag of the autocorrelation coefficient, by default 1

        Returns
        -------
        n-lag Autocorrelation coefficient: float
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_acf(self._time, self._intensity, lag=lag)

    def variance(self, biased: bool = False) -> float:
        """
        Calculate the variance of the intensity timeseries.

        Parameters
        ----------
        biased : bool, optional
            Whether to use the biased estimator of the variance, by default False.

        Returns
        -------
        variance: float
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_variance(self._time, self._intensity, biased=biased)

    def coef_variation(self, biased: bool = False) -> float:
        """
        Calculate the coefficient of variation of the intensity timeseries.

        Parameters
        ----------
        biased : bool, optional
            Whether to use the biased estimator of the standard deviation, by default False.

        Returns
        -------
        coefficient of variation: float
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_coef_var(self._time, self._intensity, biased=biased)

    def skewness(self, biased: bool = True) -> float:
        """
        Calculate the skewness of the intensity timeseries.
        The standard deviation is calculated with the biased estimator.

        Parameters
        ----------
        biased_sd : bool, optional
            Whether to use the biased estimator of the skewness, by default True.

        Returns
        -------
        skewness : float
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_skew(self._time, self._intensity, biased=biased)

    def pDry(self, threshold: float = 0) -> float:
        """
        Calculate the probability of dryness of the intensity timeseries with a threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold of dryness, by default 0

        Returns
        -------
        probability of dryness : float
        """
        if self._time.size == 0:
            return np.nan
        return _isnapshot_pDry(self._time, self._intensity, threshold=threshold)

    def rescale(self, scale: Union[float, timedelta], abs_tol: float = 1e-10) -> IndexedSnapshot:
        """
        Rescale the IndexedSnapshot timeseries to a different scale.
        New time index will be divided by the scale. And the intensity will be multiplied by the scale.
        All the time index will be divisible by the scale.
        """
        if isinstance(scale, timedelta):
            scale = scale.total_seconds() / 3600

        if self._time.size == 0:
            return type(self)()
        scale_time, scale_intensity = _isnapshot_rescale(
            self._time, self._intensity, scale, abs_tol
        )
        return type(self)(scale_time, scale_intensity)

    def unpack(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Unpack the IndexedSnapshot timeseries into a normal timeseries.
        """
        if self._time.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        else:
            if not np.all(self._time % 1 == 0):
                print(
                    "Warning: Unpacking IndexedSnapshot with float time index. Resulting time index will be rounded to integer."
                )
            diff_time = np.diff(self._time).astype(np.int64)
            intensity = np.repeat(self._intensity[:-1], diff_time)
            time = np.arange(self._time[0], self._time[-1])
            return time, intensity


def _isnapshot_check(
    time: FloatArray = None,
    intensity: FloatArray = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Check if both time and intensity are None
    if time is None or intensity is None:
        if time is not None or intensity is not None:
            raise ValueError("time and intensity must both be None or not None")

        _time: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

        _intensity: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

        return _time, _intensity

    time = np.array(time, dtype=np.float64)
    intensity = np.array(intensity, dtype=np.float64)

    # Check if time and intensity have the same length
    if time.size != intensity.size:
        raise ValueError("time and intensity must have the same length")

    # Check if time and intensity are 1D arrays
    if time.ndim != 1 or intensity.ndim != 1:
        raise ValueError("time and intensity must be 1D arrays")

    if time.size < 2:
        raise ValueError(f"time and intensity must have at least 2 elements. Got {time.size}.")

    # Check if the RLE time is strictly increasing.
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing")
    # Check if there are any nan in the time
    if np.any(np.isnan(time)):
        raise ValueError("time must not contain any nan")
    # Add ending time and intensity if there isn't one

    unique_dt = np.unique(np.diff(time))

    if len(unique_dt) == 0:
        raise ValueError("time must have at least 2 unique values")

    # Calculate count of unique time intervals
    unique_dt_count = np.bincount(np.searchsorted(unique_dt, np.diff(time)))
    # Use the most common time interval as the time interval
    time_interval = unique_dt[np.argmax(unique_dt_count)]

    if not np.isnan(intensity[-1]):
        time = np.append(time, time[-1] + time_interval)
        intensity = np.append(intensity, np.nan)

    # Check if all np.nan are at the end of the intensity timeseries
    # if (num_of_nan := np.sum(np.isnan(intensity))) >= 1:
    #    if not np.all(np.isnan(intensity[-num_of_nan:])):
    #        raise ValueError(
    #            "All np.nan must be at the end of the intensity timeseries"
    #        )

    # if num_of_nan > 1:
    #    time = time[: -num_of_nan + 1]
    #    intensity = intensity[: -num_of_nan + 1]

    # Make sure it's in IndexedSnapshot format
    # intensity_idx = np.diff(intensity, prepend=-np.inf) != 0
    intensity_idx = _index_of_first_appearance_of_consecutive_value(intensity)

    return np.array(time[intensity_idx]), np.array(intensity[intensity_idx])


@nb.njit("i8[:](f8[:])", cache=True)
def _index_of_first_appearance_of_consecutive_value(
    arr: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """
    ### This is an internal function. You shouldn't use it directly unless you know what you are doing.

    This function returns the index of the first appearance of each unique value in the array.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        1D array of float64 values.

    Returns
    -------
    index : npt.NDArray[np.int64]
        1D array of int64 values.
    """
    is_first_appearance = np.zeros_like(arr, dtype=np.int64)
    is_first_appearance[0] = 1
    last_value = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(last_value) and np.isnan(arr[i]):
            continue

        if np.isnan(last_value) and not np.isnan(arr[i]):
            is_first_appearance[i] = 1
            last_value = arr[i]
            continue

        if arr[i] != last_value:
            is_first_appearance[i] = 1
            last_value = arr[i]

    first_appearance_idx = np.where(is_first_appearance == 1)[0]

    return first_appearance_idx


def _intensity_delta_check(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if len(time) != 0:
        return np.column_stack((time, np.diff(intensity[:-1], prepend=0, append=0)))
    else:
        return np.array([], dtype=np.float64)


def _delta_to_isnapshot_archived(
    time: npt.NDArray[np.float64],
    intensity_delta: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    ### This is an internal function. You shouldn't use it directly unless you know what you are doing.

    This function convert the delta encoding of the intensity timeseries into **ALMOST** IndexedSnapshot format.
    The np.nan at the end of the intensity timeseries is not included in the IndexedSnapshot format.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows delta encoding format.
    intensity_delta : npt.NDArray[np.float64]
        1D array of intensity that follows delta encoding format.

    Returns
    -------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format (Without the np.nan at the end).
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format (Without the np.nan at the end).
    """

    # Zip time and intensity_delta into a 2D array
    delta_encoding = np.column_stack((time, intensity_delta))

    # Sort the change_time_idx by time
    delta_encoding = delta_encoding[np.argsort(delta_encoding[:, 0])]

    # Calculate the cumulative sum of the intensity changes inplace.
    np.cumsum(delta_encoding[:, 1], out=delta_encoding[:, 1])

    ## TODO: Round to 10 decimal places to avoid floating point errors. And it's inplace.
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


# Signature of two float64 arrays input and two float64 arrays output
@nb.njit("UniTuple(f8[:], 2)(f8[:], f8[:])", cache=True)
def _delta_to_isnapshot(
    time: npt.NDArray[np.float64],
    intensity_delta: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    ### This is an internal function. You shouldn't use it directly unless you know what you are doing.

    This function convert the delta encoding of the intensity timeseries into **ALMOST** IndexedSnapshot format.
    The np.nan at the end of the intensity timeseries is not included in the IndexedSnapshot format.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows delta encoding format.
    intensity_delta : npt.NDArray[np.float64]
        1D array of intensity that follows delta encoding format.

    Returns
    -------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format (Without the np.nan at the end).
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format (Without the np.nan at the end).
    """

    # Sort the change_time_idx by time
    sorted_idx = np.argsort(time)
    time = time[sorted_idx]
    intensity_delta = intensity_delta[sorted_idx]

    # Calculate the cumulative sum of the intensity changes inplace.
    for i in range(1, len(intensity_delta)):
        intensity_delta[i] += intensity_delta[i - 1]

    intensity_delta[np.where(np.abs(intensity_delta) < 1e-10)] = 0

    # Remove duplicate times by keeping the last occurence
    diff_idx = np.empty(len(time), dtype=np.bool_)
    for i in range(len(time) - 1, 0, -1):
        diff_idx[i - 1] = time[i] != time[i - 1]
    diff_idx[-1] = True

    return (
        time[diff_idx],
        intensity_delta[diff_idx],
    )


def _merge_isnapshot(a: IndexedSnapshot, b: IndexedSnapshot) -> IndexedSnapshot:
    """
    Merge two IndexedSnapshot timeseries together.

    Parameters
    ----------
    a : IndexedSnapshot
        The first IndexedSnapshot timeseries.
    b : IndexedSnapshot
        The second IndexedSnapshot timeseries.

    Returns
    -------
    IndexedSnapshot
        The merged IndexedSnapshot timeseries.
    """
    if len(a.intensity_delta) == 0 and len(b.intensity_delta) == 0:
        return IndexedSnapshot()
    if len(a.intensity_delta) == 0:
        intensity_delta = b.intensity_delta
    elif len(b.intensity_delta) == 0:
        intensity_delta = a.intensity_delta
    else:
        intensity_delta = np.concatenate((a.intensity_delta, b.intensity_delta))

    time, intensity = _delta_to_isnapshot(intensity_delta[:, 0], intensity_delta[:, 1])
    # The intensity at the end of the timeseries should be np.nan
    # But when we extract the intensity_delta, we replace it with 0 to make the calculation easier.
    # Other 0 value is included in the timeseries.
    # But the last 0 value is not included in the timeseries. So we need to add it back by replacing the last value with np.nan
    intensity[-1] = np.nan

    if len(time) == 0:
        return IndexedSnapshot()

    return IndexedSnapshot(time, intensity)


@nb.njit("UniTuple(f8, 2)(f8[:], f8[:])", cache=True)
def _isnapshot_sum_and_duration(
    time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """
    ### This is an internal function. Use `IndexedSnapshot.mean()` instead.

    Calculate the total depth of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Returns
    -------
    mean: float
        The total depth of the intensity timeseries.

    """
    # each_intensity_duration = np.diff(time)
    depth: float = 0.0
    duration: float = 0.0
    for i in range(len(time) - 1):
        if np.isnan(intensity[i]):
            continue
        depth += intensity[i] * (time[i + 1] - time[i])
        duration += time[i + 1] - time[i]

    return depth, duration


@nb.njit("f8(f8[:], f8[:])", cache=True)
def _isnapshot_mean(
    time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.mean()` instead.

    Calculate the mean of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Returns
    -------
    mean: float
        The mean of the intensity timeseries.

    """
    depth, duration = _isnapshot_sum_and_duration(time, intensity)

    if duration == 0:
        return np.nan

    return depth / duration


@nb.njit("f8(f8[:], f8[:])", cache=True)
def _isnapshot_sum_squared_error(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.standard_deviation()` instead.

    Calculate the sum of squared error of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Returns
    -------
    sum of squared error: float
        The sum of squared error of the intensity timeseries.
    """
    mean = _isnapshot_mean(time, intensity)
    sse = 0
    for i in range(len(time) - 1):
        if np.isnan(intensity[i]):
            continue
        sse += (intensity[i] - mean) ** 2 * (time[i + 1] - time[i])
    return sse


@nb.njit("f8(f8[:], f8[:], b1)", cache=True)
def _isnapshot_variance(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    biased: bool = False,
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.standard_deviation()` instead.

    Calculate the variance of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Returns
    -------
    variance: float
        The variance of the intensity timeseries.
    """
    depth, duration = _isnapshot_sum_and_duration(time, intensity)
    sse = _isnapshot_sum_squared_error(time, intensity)

    if duration - (biased is False) == 0:
        return np.nan

    return sse / (duration - (biased is False))


@nb.njit("f8(f8[:], f8[:], b1)", cache=True)
def _isnapshot_standard_deviation(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    biased: bool = False,
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.standard_deviation()` instead.

    Calculate the standard deviation of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Returns
    -------
    standard deviation: float
        The standard deviation of the intensity timeseries.
    """
    return _isnapshot_variance(time, intensity, biased) ** 0.5


@nb.njit("f8(f8[:], f8[:], b1)", cache=True)
def _isnapshot_coef_var(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    biased: bool = False,
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.standard_deviation()` instead.

    Calculate the coefficient of variation of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.
    mean : float, optional
        The mean of the intensity timeseries, If None, it will be calculated, by default None
    sse : float, optional
        The sum of squared error of the intensity timeseries, If None, it will be calculated, by default None
    biased : bool, optional
        Whether to use the biased estimator of the standard deviation, by default False.

    Returns
    -------
    coefficient of variation: float
        The coefficient of variation of the intensity timeseries.

    """
    # Coefficient of variation
    mean = _isnapshot_mean(time, intensity)

    if np.isclose(mean, 0, atol=1e-15):
        return np.nan

    return _isnapshot_standard_deviation(time, intensity, biased) / mean


@nb.njit("f8(f8[:], f8[:], b1)", cache=True)
def _isnapshot_skew(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    biased: bool = False,
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.standard_deviation()` instead.

    Calculate the skewness of the intensity timeseries.
    The standard deviation is calculated with the unbiased estimator.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Returns
    -------
    skewness : float
        The skewness of the intensity timeseries.
    """
    # Skewness
    depth, duration = _isnapshot_sum_and_duration(time, intensity)

    if duration == 0:
        return np.nan

    mean = depth / duration

    stddev = _isnapshot_standard_deviation(time, intensity, biased=True)
    if np.isclose(stddev, 0, atol=1e-15):
        return np.nan

    skewness = 0
    for i in range(len(time) - 1):
        if np.isnan(intensity[i]):
            continue
        skewness += (intensity[i] - mean) ** 3 * (time[i + 1] - time[i])
    skewness = skewness / (duration * stddev**3)

    if biased is False:
        skewness *= (duration * (duration - 1)) ** 0.5 / (duration - 2)
    return skewness


def _isnapshot_pDry(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    threshold: float = 0,
) -> float:
    """
    ### This is an internal function. Use `IndexedSnapshot.pDry()` instead.

    Calculate the probability of dryness of the intensity timeseries with a threshold.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.
    threshold : float, optional
        The threshold of dryness, by default 0

    Returns
    -------
    probability of dryness : float
    """
    duration = 0
    wet_duration = 0
    for i in range(len(time) - 1):
        if np.isnan(intensity[i]):
            continue
        duration += time[i + 1] - time[i]
        if intensity[i] > threshold:
            wet_duration += time[i + 1] - time[i]

    if duration == 0:
        return np.nan

    return 1 - (wet_duration / duration)


@nb.njit("f8(f8[:], f8[:], f8)", cache=True)
def _isnapshot_acf(
    time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    lag: Optional[float] = 1,
):
    """
    ### This is an internal function. Use `IndexedSnapshot.acf()` instead.

    Calculate the n-lag autocorrelation coefficient of the intensity timeseries.
    It's basically Cov(X_t, X_{t+lag}) / Var(X_t)

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.
    lag : int, optional
        The lag of the autocorrelation coefficient, by default 1

    Returns
    -------
    n-lag Autocorrelation coefficient: float
        The n-lag autocorrelation coefficient of the intensity timeseries.
    """
    if lag == 0:
        return 1
    n = len(time)

    mean = _isnapshot_mean(time, intensity)

    sse = _isnapshot_sum_squared_error(time, intensity)

    if np.isclose(sse, 0, atol=1e-15):
        return np.nan

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
            if np.isnan(intensity[x_idx - 1]) or np.isnan(intensity[y_idx - 1]):
                x_idx += 1
                continue
            size = (time[x_idx] - lag) - max(time[y_idx - 1], (time[x_idx - 1] - lag))
            result += (
                (intensity[y_idx - 1] - mean) * (intensity[x_idx - 1] - mean)
            ) * size
            x_idx += 1
        else:
            if np.isnan(intensity[x_idx - 1]) or np.isnan(intensity[y_idx - 1]):
                y_idx += 1
                continue
            size = time[y_idx] - max(time[y_idx - 1], (time[x_idx - 1] - lag))
            result += (
                (intensity[y_idx - 1] - mean) * (intensity[x_idx - 1] - mean)
            ) * size
            y_idx += 1

    return result / sse

@nb.njit("f8(f8, f8)", cache=True)
def mod(a: float, b: float) -> float:
    return a - b * np.floor(a / b)

@nb.njit("UniTuple(f8[:], 2)(f8[:], f8[:], f8, f8)", cache=True)
def _isnapshot_rescale(
    time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], scale: float, abs_tol: float = 1e-10
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    ### This is an internal function. Use `IndexedSnapshot.standard_deviation()` instead.

    Rescale the IndexedSnapshot timeseries to a different scale.
    New time index will be divided by the scale. And the intensity will be multiplied by the scale.
    All the time index will be divisible by the scale.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.
    scale : float
        The scale to be rescaled to.
    abs_tol : float, optional
        The absolute tolerance for floating point error, by default 1e-10

    Returns
    -------
    scaled time : npt.NDArray[np.float64]
        1D array of time index that follows IndexedSnapshot format.
    scaled intensity : npt.NDArray[np.float64]
        1D array of intensity that follows IndexedSnapshot format.

    Examples
    --------
    >>> time = np.array([0, 10, 13, 18, 33])
    >>> intensity = np.array([0, 3, 0, 5, np.nan])
    >>> scale = 5
    >>> _isnapshot_rescale(time, intensity, scale)
    (array([0., 2., 3., 4., 6., 7.]), array([ 0.,  9., 10., 25., 15., nan]))
    """
    time_index = len(time) - 1
    scale_time = np.full(time_index * 3 + 2, np.nan)
    scale_intensity = np.full(time_index * 3 + 2, np.nan)
    #scale_time = np.zeros(time_index * 3 + 2)
    #scale_intensity = np.zeros(time_index * 3 + 2)

    scale_time[0] = np.nan
    rescale_idx = 1
    for i in range(time_index):
        srt, end = time[i], time[i + 1]
        r_srt, r_end = np.floor(srt / scale), np.floor(end / scale)
        intensity_i = intensity[i]

        # Case 1 might overshot. Case 3 and Case 2 almost always overshot. As you can see in their blow diagram.
        # But we still move to the next rescale time index. So we need to drop back to the previous rescale time index.
        if scale_time[rescale_idx - 1] == r_srt:
            rescale_idx -= 1

        # Case 1: When two time index are in the same rescale range.
        # rescale:      |-------------------------------------|
        #                   |  A  |  X  |  X  |  ...
        # original:         |-----|-----|-----|  ...
        if r_srt == r_end:
            scale_time[rescale_idx] = r_srt
            if np.isnan(intensity_i):
                pass
            elif np.isnan(scale_intensity[rescale_idx]):
                scale_intensity[rescale_idx] = intensity_i * (end - srt)
            else:
                scale_intensity[rescale_idx] += intensity_i * (end - srt)
            rescale_idx += 1
            continue

        # Case 2: When two time index cross "1" rescale range.
        # rescale:    |-----------|-----------|
        #                 |   A   |  B  |
        # original:       |-------------|
        if r_srt + 1 == r_end:
            scale_time[rescale_idx] = r_srt
            if np.isnan(intensity_i):
                pass
            elif np.isnan(scale_intensity[rescale_idx]):
                scale_intensity[rescale_idx] = intensity_i * (scale - mod(srt, scale))
            else:
                scale_intensity[rescale_idx] += intensity_i * (scale - mod(srt, scale))
            rescale_idx += 1

            if np.isclose(mod(end, scale), 0, atol=1e-10):
                # It means that the end time is the same as the next rescale time.
                # We shouldn't add the intensity 0 to the next rescale time.
                # Because it might make nan intensity to be 0. Which is wrong.
                continue

            scale_time[rescale_idx] = r_end
            if np.isnan(intensity_i):
                pass
            elif np.isnan(scale_intensity[rescale_idx]):
                scale_intensity[rescale_idx] = intensity_i * mod(end, scale)
            else:
                scale_intensity[rescale_idx] += intensity_i * mod(end, scale)
            rescale_idx += 1
            continue

        # Case 3: When two time index cross multiple rescale range. We can finish all the middle rescale range (B).
        # rescale:    |-----------|------------|-----------|
        #                     | A |     B      | C |
        # original:           |--------------------|
        if r_srt + 1 < r_end:
            scale_time[rescale_idx] = r_srt
            if np.isnan(intensity_i):
                pass
            elif np.isnan(scale_intensity[rescale_idx]):
                scale_intensity[rescale_idx] = intensity_i * (scale - mod(srt, scale))
            else:
                scale_intensity[rescale_idx] += intensity_i * (scale - mod(srt, scale))
            rescale_idx += 1

            scale_time[rescale_idx] = r_srt + 1
            if np.isnan(intensity_i):
                pass
            elif np.isnan(scale_intensity[rescale_idx]):
                scale_intensity[rescale_idx] = intensity_i * scale
            else:
                scale_intensity[rescale_idx] += intensity_i * scale
            rescale_idx += 1

            if np.isclose(mod(end, scale), 0, atol=1e-10):
                # It means that the end time is the same as the next rescale time.
                # We shouldn't add the intensity 0 to the next rescale time.
                # Because it might make nan intensity to be 0. Which is wrong.
                continue

            scale_time[rescale_idx] = r_end
            if np.isnan(intensity_i):
                pass
            elif np.isnan(scale_intensity[rescale_idx]):
                scale_intensity[rescale_idx] = intensity_i * mod(end, scale)
            else:
                scale_intensity[rescale_idx] += intensity_i * mod(end, scale)
            rescale_idx += 1
            continue

    # Append the ending time with nan intensity
    # If the last rescaled time is the same as the last original time, change it to nan.

    scale_time = scale_time[1 : rescale_idx + 1]
    scale_time[-1] = np.ceil(time[-1] / scale)
    scale_intensity = scale_intensity[1 : rescale_idx + 1]

    # Preallocate boolean array
    diff_time = np.ones(len(scale_time), dtype=np.bool_)
    diff_time[0] = True
    #round_digit = int(-np.log10(abs_tol))
    for i in range(1, len(scale_time)):
        # Skip current time index if it's the same as the previous one.
        ## Current intensity is the same as the previous one.
        if scale_intensity[i] == scale_intensity[i - 1]:
            diff_time[i] = False
            continue
        ## Current intensity is close to the previous one. (Floating point error)
        if np.isclose(scale_intensity[i], scale_intensity[i - 1], rtol=abs_tol):
            diff_time[i] = False
            continue
        ## Current intensity is the same as the previous one. (Both are np.nan)
        if np.isnan(scale_intensity[i]) and np.isnan(scale_intensity[i - 1]):
            diff_time[i] = False
            continue

        # Keep the current time index if it's different from the previous one.
        ## Current intensity is np.nan.
        if np.isnan(scale_intensity[i]):
            continue

        ## Current intensity is float. Round it to atol.
        #scale_intensity[i] = np.round(scale_intensity[i], round_digit)

    # Replace all np.nan in scale_intensity to 0
    #scale_intensity[np.isnan(scale_intensity)] = 0

    return scale_time[diff_time], scale_intensity[diff_time]
