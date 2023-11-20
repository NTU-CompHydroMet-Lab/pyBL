from __future__ import annotations

from typing import (
    Any,
    List,
    Optional,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt
import numba as nb  # type: ignore

__all__ = ["IntensityMRLE"]

IMRLESequence = Optional[Union[npt.NDArray[np.float64], npt.NDArray[np.int64], List[float]]]


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
        if len(self._time) != 0:
            self._intensity_delta: npt.NDArray[np.float64] = np.column_stack(
                (self._time, np.diff(self._intensity[: -1], prepend=0, append=0))
            )
        else:
            self._intensity_delta = np.array([], dtype=np.float64)
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
            Duplicate times will be summed together. (Because it's multiple delta at the same time)

        intensity_delta: Optional[Union[`np.ndarray`, `List`]]
            Delta Encoded Intensity values (mm/h) of the intensity timeseries.

        """
        time, intensity_mrle = _delta_to_mrle_nb(time, intensity_delta)

        return cls(time, intensity_mrle, scale)

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return self._time

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        return self._intensity

    @property
    def intensity_delta(self) -> npt.NDArray[np.float64]:
        return self._intensity_delta

    def add(self, timeseries: IntensityMRLE, sequential: bool=False) -> None:
        if len(timeseries.time) == 0:
            return
        if sequential is True:
            if len(self._time) != 0:
                # Shift the start of the timeseries to be right after th last time of the current timeseries
                timeseries = IntensityMRLE(
                    timeseries.time - (timeseries.time[0] - self._time[-1]), timeseries.intensity, timeseries._scale
                )
        self.__iadd__(timeseries)

    def __iadd__(self, timeseries: IntensityMRLE) -> IntensityMRLE:

        result_mrle = _merge_mrle(self, timeseries)

        self._time, self._intensity = result_mrle._time, result_mrle._intensity
        self._intensity_delta = result_mrle._intensity_delta

        return self

    def __add__(self, timeseries: IntensityMRLE) -> IntensityMRLE:
        return _merge_mrle(self, timeseries)

    @overload
    def __getitem__(self, time_idx: slice) -> IntensityMRLE:
        ...

    @overload
    def __getitem__(self, time_idx: int) -> np.float64:
        ...

    def __getitem__(self, time_idx: Any) -> Any:
        if isinstance(time_idx, slice):
            return self._get_slice_idx(time_idx)
        elif isinstance(time_idx, int):
            return self._get_int_idx(time_idx)
        else:
            raise TypeError("time_idx must be an int or a slice")

    def _get_slice_idx(self, time_idx: slice) -> IntensityMRLE:
        if time_idx.start >= time_idx.stop:
            return type(self)(scale=self._scale)
        # Use binary search to find the index
        start_idx = np.searchsorted(a=self._time, v=time_idx.start, side="right")
        stop_idx = np.searchsorted(a=self._time, v=time_idx.stop, side="left")

        if start_idx == len(self._time):
            return type(self)(scale=self._scale)
        if stop_idx == 0:
            return type(self)(scale=self._scale)

        time = self._time[max(0, start_idx-1):stop_idx]
        intensity = self._intensity[max(0, start_idx-1):stop_idx]

        if start_idx != 0:
            time[0] = time_idx.start
        if stop_idx < len(self._time) and time[-1] != time_idx.stop-1:
            time = np.append(time, time_idx.stop-1)
            intensity = np.append(intensity, intensity[-1])

        return type(self)(time, intensity, self._scale)

    def _get_int_idx(self, time_idx: int) -> np.float64:
        # Use binary search to find the index
        idx = np.searchsorted(a=self._time, v=time_idx, side="right")
        if idx == 0 or idx == len(self._time):
            return np.float64(0.0)
        return self._intensity[int(idx) - 1]

    def __str__(self) -> str:
        time_value = "\n".join(f'{self.time[i]:>5.7f} {self.intensity[i]:>5.7f}' for i in range(len(self.time)))
        return time_value
    
    def total(self) -> float:
        '''
        Calculate the total depth of the intensity timeseries.
        '''
        if self._time.size == 0:
            return 0
        return _mrle_total(self._time, self._intensity)

    def mean(self) -> Union[float]:
        '''
        Calculate the mean of the intensity timeseries.
        '''
        if self._time.size == 0:
            return np.nan
        return _mrle_mean(self._time, self._intensity)

    def acf(self, lag=1) -> float:
        '''
        Calculate the n-lag autocorrelation coefficient of the intensity timeseries.
        It's basically Cov(X_t, X_{t+lag}) / Var(X_t)

        Parameters
        ----------
        lag : int, optional
            The lag of the autocorrelation coefficient, by default 1
        
        Returns
        -------
        n-lag Autocorrelation coefficient: float
        '''
        if self._time.size == 0:
            return np.nan
        return _mrle_acf(self._time, self._intensity, lag=lag)

    def cvar(self, biased=False) -> float:
        '''
        Calculate the coefficient of variation of the intensity timeseries.

        Parameters
        ----------
        biased : bool, optional
            Whether to use the biased estimator of the standard deviation, by default False.

        Returns
        -------
        coefficient of variation: float
        '''
        if self._time.size == 0:
            return np.nan
        return _mrle_coef_var(self._time, self._intensity, biased=biased)

    def skewness(self, biased=True) -> float:
        '''
        Calculate the skewness of the intensity timeseries.
        The standard deviation is calculated with the biased estimator.
        
        Parameters
        ----------
        biased_sd : bool, optional
            Whether to use the biased estimator of the skewness, by default True.
        
        Returns
        -------
        skewness : float
        '''
        if self._time.size == 0:
            return np.nan
        return _mrle_skew(self._time, self._intensity, biased=biased)

    def pDry(self, threshold: float=0) -> float:
        '''
        Calculate the probability of dryness of the intensity timeseries with a threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold of dryness, by default 0
        
        Returns
        -------
        probability of dryness : float
        '''
        if self._time.size == 0:
            return np.nan
        return _mrle_pDry(self._time, self._intensity, threshold=threshold)

    def rescale(self, scale: float) -> IntensityMRLE:
        '''
        Rescale the MRLE timeseries to a different scale.
        New time index will be divided by the scale. And the intensity will be multiplied by the scale.
        All the time index will be divisible by the scale.
        '''
        if self._time.size == 0:
            return type(self)(scale=self._scale * scale)
        scale_time, scale_intensity = _mrle_rescale(self._time, self._intensity, scale)
        return type(self)(scale_time, scale_intensity, self._scale * scale)

    def unpack(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        '''
        Unpack the MRLE timeseries into a normal timeseries.
        '''
        if self._time.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        else:
            if not np.all(self._time % 1 == 0):
                print("Warning: Unpacking MRLE with float time index. Resulting time index will be rounded to integer.")
            diff_time = np.diff(self._time).astype(np.int64)
            intensity = np.repeat(self._intensity[:-1], diff_time)
            time = np.arange(self._time[0], self._time[-1])
            return time, intensity



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

def _delta_to_mrle(
    time: IMRLESequence = None,
    intensity_delta: IMRLESequence = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    '''
    ### This is an internal function. You shouldn't use it directly unless you know what you are doing.  

    This function convert the delta encoding of the intensity timeseries into **ALMOST** MRLE format.  
    The np.nan at the end of the intensity timeseries is not included in the MRLE format.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows delta encoding format.
    intensity_delta : npt.NDArray[np.float64]
        1D array of intensity that follows delta encoding format.
    
    Returns
    -------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format (Without the np.nan at the end).
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format (Without the np.nan at the end).  
    '''

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
@nb.njit('UniTuple(f8[:], 2)(f8[:], f8[:])')
def _delta_to_mrle_nb(
    time: IMRLESequence = None,
    intensity_delta: IMRLESequence = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Sort the change_time_idx by time
    sorted_idx = np.argsort(time)
    time = time[sorted_idx]
    intensity_delta = intensity_delta[sorted_idx]

    # Calculate the cumulative sum of the intensity changes inplace.
    for i in range(1, len(intensity_delta)):
        intensity_delta[i] += intensity_delta[i-1]

    ## TODO: Round to 10 decimal places to avoid floating point errors. And it's inplace.
    #delta_encoding.round(10, out=delta_encoding)

    # Remove duplicate times by keeping the last occurence
    diff_idx = np.empty(len(time), dtype=np.bool_)
    for i in range(len(time)-1, 0, -1):
        diff_idx[i-1] = time[i] != time[i-1]
    diff_idx[-1] = True

    return (
        time[diff_idx],
        intensity_delta[diff_idx],
    )

def _merge_mrle(a: IntensityMRLE, b: IntensityMRLE) -> IntensityMRLE:
    '''
    Merge two MRLE timeseries together.

    Parameters
    ----------
    a : IntensityMRLE
        The first IntensityMRLE timeseries.
    b : IntensityMRLE
        The second IntensityMRLE timeseries.
    
    Returns
    -------
    IntensityMRLE
        The merged IntensityMRLE timeseries.
    '''
    if a._scale != b._scale:
        raise ValueError("Merging two timeseries with different scale is not supported '''YET'''.")
    if len(a.intensity_delta) == 0 and len(b.intensity_delta) == 0:
        return IntensityMRLE(scale=a._scale)
    if len(a.intensity_delta) == 0:
        intensity_delta = b.intensity_delta
    elif len(b.intensity_delta) == 0:
        intensity_delta = a.intensity_delta
    else:
        intensity_delta = np.concatenate((a.intensity_delta, b.intensity_delta))

    time, intensity= _delta_to_mrle(intensity_delta[:, 0], intensity_delta[:, 1])
    # The intensity at the end of the timeseries should be np.nan
    # But when we extract the intensity_delta, we replace it with 0 to make the calculation easier.
    # Other 0 value is included in the timeseries.
    # But the last 0 value is not included in the timeseries. So we need to add it back by replacing the last value with np.nan
    intensity[-1] = np.nan

    if len(time) == 0:
        return IntensityMRLE(scale=a._scale)

    return IntensityMRLE(time, intensity, a._scale)

@nb.njit
def _mrle_total(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.mean()` instead.

    Calculate the total depth of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.

    Returns
    -------
    mean: float
        The total depth of the intensity timeseries. 
        
    '''
    #each_intensity_duration = np.diff(time)
    total = 0
    for i in range(len(time)-1):
        total += intensity[i] * (time[i+1] - time[i])
    return total

@nb.njit    
def _mrle_mean(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.mean()` instead.

    Calculate the mean of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.

    Returns
    -------
    mean: float
        The mean of the intensity timeseries.
        
    '''
    return _mrle_total(time, intensity) / (time[-1] - time[0])


@nb.njit
def _mrle_sum_squared_error(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.standard_deviation()` instead.

    Calculate the sum of squared error of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    mean : float, optional
        The mean of the intensity timeseries, If None, it will be calculated, by default None

    Returns
    -------
    sum of squared error: float
        The sum of squared error of the intensity timeseries.
    '''
    if mean is None:
        mean = _mrle_mean(time, intensity)
    sse = 0
    for i in range(len(time)-1):
        sse += (intensity[i] - mean)**2 * (time[i+1] - time[i])
    return sse

@nb.njit
def _mrle_variance(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None, sse = None, biased=False) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.standard_deviation()` instead.

    Calculate the variance of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    mean : float, optional
        The mean of the intensity timeseries, If None, it will be calculated, by default None
    sse : float, optional
        The sum of squared error of the intensity timeseries, If None, it will be calculated, by default None
    biased : bool, optional
        Whether to use the biased estimator of the variance, by default False.
        
    Returns
    -------
    variance: float
        The variance of the intensity timeseries.
    '''
    if mean is None:
        mean = _mrle_mean(time, intensity)
    if sse is None:
        sse = _mrle_sum_squared_error(time, intensity, mean)
    return sse / (time[-1] - time[0] - (biased is False))

@nb.njit
def _mrle_standard_deviation(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None, sse = None, biased = False) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.standard_deviation()` instead.

    Calculate the standard deviation of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    mean : float, optional
        The mean of the intensity timeseries, If None, it will be calculated, by default None
    sse : float, optional
        The sum of squared error of the intensity timeseries, If None, it will be calculated, by default None
    biased : bool, optional
        Whether to use the biased estimator of the standard deviation, by default False.

    Returns
    -------
    standard deviation: float
        The standard deviation of the intensity timeseries.
    '''
    if mean is None:
        mean = _mrle_mean(time, intensity)
    if sse is None:
        sse = _mrle_sum_squared_error(time, intensity, mean)
    return (_mrle_variance(time, intensity, mean, sse, biased))**0.5

@nb.njit
def _mrle_coef_var(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None, sse = None, biased = False) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.standard_deviation()` instead.

    Calculate the coefficient of variation of the intensity timeseries.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
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

    '''
    # Coefficient of variation
    if mean is None:
        mean = _mrle_mean(time, intensity)

    if mean == 0:
        return np.nan

    if sse is None:
        sse = _mrle_sum_squared_error(time, intensity, mean)

    return _mrle_standard_deviation(time, intensity, mean=mean, sse=sse, biased=biased) / mean

@nb.njit
def _mrle_skew(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], mean = None, sd = None, biased = False) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.standard_deviation()` instead.

    Calculate the skewness of the intensity timeseries.
    The standard deviation is calculated with the unbiased estimator.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    mean : float, optional
        The mean of the intensity timeseries, If None, it will be calculated, by default None
    sd : float, optional
        The standard deviation of the intensity timeseries, If None, it will be calculated, by default None
    biased : bool, optional
        Whether to use the biased estimator of the skewness, by default False.
        If False, the unbiased estimator will be used. Which is (n*(n-1))**0.5 / (n-2) * biased_skewness
    
    Returns
    -------
    skewness : float
        The skewness of the intensity timeseries.
    '''
    # Skewness
    n = time[-1] - time[0]
    if mean is None:
        mean = _mrle_mean(time, intensity)

    if sd is None:
        sd = _mrle_standard_deviation(time, intensity, mean, biased=False)

    # Skewness
    skewness = 0
    for i in range(len(time)-1):
        skewness += (intensity[i] - mean)**3 * (time[i+1] - time[i])
    skewness = skewness/(((time[-1] - time[0]))*sd**3)

    if biased is False:
        skewness *= (n*(n-1))**0.5 / (n-2)
    return skewness

def _mrle_pDry(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], threshold: float=0) -> float:
    '''
    ### This is an internal function. Use `IntensityMRLE.pDry()` instead.

    Calculate the probability of dryness of the intensity timeseries with a threshold.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    threshold : float, optional
        The threshold of dryness, by default 0

    Returns
    -------
    probability of dryness : float
    '''
    wet_time = 0
    for i in range(len(time)-1):
        if intensity[i] > threshold:
            wet_time += time[i+1] - time[i]

    return 1 - (wet_time / (time[-1] - time[0]))


@nb.njit
def _mrle_acf(time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], lag=1, mean = None, sse = None):
    '''
    ### This is an internal function. Use `IntensityMRLE.acf()` instead.

    Calculate the n-lag autocorrelation coefficient of the intensity timeseries.  
    It's basically Cov(X_t, X_{t+lag}) / Var(X_t)  

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    lag : int, optional
        The lag of the autocorrelation coefficient, by default 1
    mean : float, optional
        The mean of the intensity timeseries, If None, it will be calculated, by default None
    sse : float, optional
        The sum of squared error of the intensity timeseries, If None, it will be calculated, by default None

    Returns
    -------
    n-lag Autocorrelation coefficient: float
        The n-lag autocorrelation coefficient of the intensity timeseries.
    '''
    if lag == 0:
        return 1
    n = len(time)

    if mean is None:
        mean = _mrle_mean(time, intensity)

    if sse is None:
        sse = _mrle_sum_squared_error(time, intensity, mean)
    
    if sse == 0:
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
            size = (time[x_idx] - lag) - max(time[y_idx-1], (time[x_idx-1]-lag))
            result += ((intensity[y_idx-1]-mean) * (intensity[x_idx-1]-mean)) * size
            x_idx += 1
        else:
            size = time[y_idx] - max(time[y_idx-1], (time[x_idx-1]-lag))
            result += ((intensity[y_idx-1]-mean) * (intensity[x_idx-1]-mean)) * size
            y_idx += 1
    return result / sse

@nb.njit()
def _mrle_rescale(
    time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64], scale: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    '''
    ### This is an internal function. Use `IntensityMRLE.standard_deviation()` instead.

    Rescale the MRLE timeseries to a different scale.
    New time index will be divided by the scale. And the intensity will be multiplied by the scale.
    All the time index will be divisible by the scale.

    Parameters
    ----------
    time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
    scale : float
        The scale to be rescaled to.
    
    Returns
    -------
    scaled time : npt.NDArray[np.float64]
        1D array of time index that follows MRLE format.
    scaled intensity : npt.NDArray[np.float64]
        1D array of intensity that follows MRLE format.
            
    Examples
    --------
    >>> time = np.array([0, 10, 13, 18, 33])
    >>> intensity = np.array([0, 3, 0, 5, np.nan])
    >>> scale = 5
    >>> _mrle_rescale(time, intensity, scale)
    (array([0., 2., 3., 4., 6., 7.]), array([ 0.,  9., 10., 25., 15., nan]))
    '''
    time_index = len(time) - 1
    scale_time = np.zeros(time_index*3+2)
    scale_intensity = np.zeros(time_index*3+2)

    scale_time[0] = np.nan
    rescale_idx = 1
    for i in range(time_index):
        srt, end = time[i], time[i+1]
        r_srt, r_end = srt//scale, end//scale
        intensity_i = intensity[i]

        if scale_time[rescale_idx - 1] == r_srt:
            rescale_idx -= 1

        #original:      |------------|
        #rescale:            |----|
        if r_srt == r_end:
            scale_time[rescale_idx] = r_srt
            scale_intensity[rescale_idx] += intensity_i*(end-srt)
            rescale_idx += 1

        #original:      |------------|
        #           | A |  B  |
        #rescale:   |---------|
        if r_srt + 1 == r_end:
            scale_time[rescale_idx] = r_srt
            scale_intensity[rescale_idx] += intensity_i*(scale - srt % scale)
            rescale_idx += 1
            scale_time[rescale_idx] = r_end
            scale_intensity[rescale_idx] += intensity_i*(end % scale)
            rescale_idx += 1

        #original:      |------------|
        #           | A |     B      | C |
        #rescale:   |--------------------|
        if r_srt + 1 < r_end:
            scale_time[rescale_idx] = r_srt
            scale_intensity[rescale_idx] += intensity_i*(scale - srt % scale)
            rescale_idx += 1
            scale_time[rescale_idx] = r_srt + 1
            scale_intensity[rescale_idx] += intensity_i*scale
            rescale_idx += 1
            scale_time[rescale_idx] = r_end
            scale_intensity[rescale_idx] += intensity_i*(end % scale)
            rescale_idx += 1

    # Append the ending time with nan intensity
    # If the last rescaled time is the same as the last original time, change it to nan.
    if scale_time[rescale_idx - 1] == time[-1]/scale:
        scale_intensity[rescale_idx - 1] = np.nan
        rescale_idx -= 1
    else:
        scale_time[rescale_idx] = scale_time[rescale_idx - 1] + 1
        scale_intensity[rescale_idx] = np.nan
    
    scale_time = scale_time[1:rescale_idx+1]
    scale_intensity = scale_intensity[1:rescale_idx+1]

    # Preallocate boolean array
    diff_time = np.ones(len(scale_time), dtype=np.bool_)
    for i in range(1, len(scale_time)):
        diff_time[i] = (scale_intensity[i] != scale_intensity[i-1])
    diff_time[0] = True

    return scale_time[diff_time] , scale_intensity[diff_time]

