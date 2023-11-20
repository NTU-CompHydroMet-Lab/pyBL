from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore

from pyBL.timeseries import IntensityMRLE


def get_month_intervals(
    start: Optional[Union[int, datetime]] = None,
    end: Optional[Union[int, datetime]] = None,
) -> npt.NDArray[np.float64]:
    """
    ### Note that the returned unix time is `UTC`.
    Generate a 2D matrix of (nYear, 12, 2) of unix time interval of each month with its start and exclusive end unix time.

    Parameters
    ----------
    start : int, datetime, None
        Start Year, default to be 1900
    end : int, datetime, None
        End Year, default to be 2100

    Returns
    -------
    np.ndarray
        2D matrix of (nYear, 12, 2) of unix time interval of each month with its start and exclusive end unix time.
    """
    if isinstance(start, int):
        start = datetime(start, 1, 1, 0, 0, 0)
    if isinstance(end, int):
        end = datetime(end, 1, 1, 0, 0, 0)

    if start is None:
        start = datetime(1900, 1, 1, 0, 0, 0)
    if end is None:
        end = datetime(2100, 1, 1, 0, 0, 0)
    day = pd.date_range(start=start, end=end, freq="MS")
    # Convert to unix time
    month_srt = day.astype("int64") // 10**9
    month_end = month_srt
    # Stack them together
    month_intervals = np.stack((month_srt[:-1], month_end[1:]), axis=1)
    # Group the month_interval by month
    month_intervals = np.reshape(month_intervals, (-1, 12, 2))
    return month_intervals


def to_year_month(
    unix_time: npt.NDArray[np.float64], intensity: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Convert input 1D array of unix time and intensity to 2D matrix of year and month.
    With the shape of (nYear, 12). Each cell in the 2D matrix is a `IntensityMRLE` object of that month.

    Parameters
    ----------
    unix_time : npt.NDArray
        1D array of unix time. Unit: second
    intensity : npt.NDArray
        1D array of intensity data. Unit: mm/h

    Returns
    -------
    npt.NDArray
        2D matrix of year and month full of `IntensityMRLE` objects.
        With the shape of (nYear, 12).
    """
    mrle = IntensityMRLE(unix_time, intensity / 3600)
    # Gest starting year and ending year
    start_year = datetime.fromtimestamp(unix_time[0], tz=timezone.utc).year
    end_year = datetime.fromtimestamp(unix_time[-1], tz=timezone.utc).year
    month_interval_each_year = get_month_intervals(start=start_year, end=end_year)
    mrleYearMonth = np.empty(
        (len(month_interval_each_year), 12), dtype=IntensityMRLE
    )  # (year, month)
    for i, year in enumerate(month_interval_each_year):
        for j, month in enumerate(year):
            mrleYearMonth[i, j] = mrle[month[0] : month[1]]

    # Find the first row that doesn't have any IntensityMRLE object
    for i, row in enumerate(mrleYearMonth):
        if np.isnan(row[0].mean()):
            break

    return mrleYearMonth


def preprocess_classic(
    unix_time: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    timescale: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    This function calculate the statistics of the input timeseries and return the target and weight matrix for fitting.

    Parameters
    ----------
    unix_time : np.ndarray
        1D array of unix time. Unit: second

    intensity : np.ndarray
        1D array of intensity data. Unit: mm/h

    timescale : np.ndarray
        1D array of timescale. Unit: hour

    Returns
    -------
    target : np.ndarray
        3D matrix of target. With the shape of (12 (month), nScale, nStats).

    weight : np.ndarray
        3D matrix of weight. With the shape of (12 (month), nScale, nStats).
    """
    timescale = np.array(timescale)
    timescale = timescale * 3600  # Convert to second

    mrle = IntensityMRLE(
        unix_time, intensity / 3600
    )  # Unit: mm/h so divide by 3600 to get mm/s
    month_interval_each_year = get_month_intervals()
    # Segment the mrle timeseries into months from 1900 to 2100
    mrle_month_each = np.empty(
        (12, len(month_interval_each_year), len(timescale)), dtype=IntensityMRLE
    )  # (month, year, scale)
    for i, year in enumerate(month_interval_each_year):
        for j, month in enumerate(year):
            for k, scale in enumerate(timescale):
                mrle_month_each[j, i, k] = mrle[month[0] : month[1]].rescale(scale)

    # MRLE that stores the total of each month
    mrle_month_total = np.empty(
        (12, len(timescale)), dtype=IntensityMRLE
    )  # (month, scale)
    for i in range(12):
        for j in range(len(mrle_month_each[0])):
            for k, scale in enumerate(timescale):
                if j == 0:
                    mrle_month_total[i, k] = IntensityMRLE(scale=scale)
                mrle_month_total[i, k].add(mrle_month_each[i, j, k], sequential=True)

    stats_month = np.zeros((12, len(timescale), 5))  # (month, scale, stats)
    for month in range(12):
        for scale in range(len(timescale)):
            model = mrle_month_total[month, scale]
            stats_month[month, scale, :] = [
                model.mean(),
                model.cvar(),
                model.acf(1),
                model.skewness(),
                model.pDry(0),
            ]

    stats_month_seperate = np.zeros(
        (12, len(month_interval_each_year), len(timescale), 5)
    )  # (month, year, scale, stats)
    for month in range(12):
        for year in range(len(month_interval_each_year)):
            for scale in range(len(timescale)):
                model = mrle_month_each[month, year, scale]
                stats_month_seperate[month, year, scale, :] = [
                    model.mean(),
                    model.cvar(),
                    model.acf(1),
                    model.skewness(),
                    model.pDry(0),
                ]

    stats_weight = 1 / np.nanvar(stats_month_seperate, axis=1)  # (month, scale, stats)

    return stats_month, stats_weight
