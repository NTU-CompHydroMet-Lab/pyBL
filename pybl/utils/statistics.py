from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore

from pybl.models import StatMetrics
from pybl.timeseries import IndexedSnapshot


def get_month_intervals(
    start: Optional[Union[int, datetime]] = None,
    end: Optional[Union[int, datetime]] = None,
) -> npt.NDArray[np.float64]:
    """Generate a 2D matrix of unix time intervals for each month.

    The returned unix timestamps are in UTC.

    Parameters
    ----------
    start : int or datetime, optional
        Start year. Defaults to 1900.
    end : int or datetime, optional
        End year. Defaults to 2100.

    Returns
    -------
    intervals : np.ndarray
        Array of shape (nYear, 12, 2) containing the start and exclusive end
        unix timestamps for each month.
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
    """Convert 1D arrays of unix time and intensity to a 2D year-month matrix.

    Parameters
    ----------
    unix_time : np.ndarray
        1D array of unix timestamps in seconds.
    intensity : np.ndarray
        1D array of intensity values in mm/h.

    Returns
    -------
    series_year_month : np.ndarray
        Array of shape (nYear, 12) where each cell is an
        `IndexedSnapshot` for that month.
    """
    series = IndexedSnapshot(unix_time, intensity / 3600)
    # Gest starting year and ending year
    start_year = datetime.fromtimestamp(unix_time[0], tz=timezone.utc).year
    end_year = datetime.fromtimestamp(unix_time[-1], tz=timezone.utc).year
    month_interval_each_year = get_month_intervals(start=start_year, end=end_year)
    seriesYearMonth = np.empty(
        (len(month_interval_each_year), 12), dtype=IndexedSnapshot
    )  # (year, month)
    for i, year in enumerate(month_interval_each_year):
        for j, month in enumerate(year):
            seriesYearMonth[i, j] = series[month[0] : month[1]]

    # Find the first row that doesn't have any IndexedSnapshot object
    for i, row in enumerate(seriesYearMonth):
        if np.isnan(row[0].mean()):
            break

    return seriesYearMonth


def calculate_stats_matrix(
    timeindex: npt.NDArray[np.float64],
    intensity: npt.NDArray[np.float64],
    timescale: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate statistics of the input timeseries for model fitting.

    Parameters
    ----------
    timeindex : np.ndarray
        1D array of unix timestamps in seconds.
    intensity : np.ndarray
        1D array of intensity values in mm/s.
    timescale : np.ndarray
        1D array of timescales in seconds.

    Returns
    -------
    target : np.ndarray
        Shape (12, nScale, nStats) target matrix.
    weight : np.ndarray
        Shape (12, nScale, nStats) weight matrix.
    """
    timescale = np.array(timescale)

    series = IndexedSnapshot(
        timeindex, intensity
    )  # Unit: mm/h so divide by 3600 to get mm/s
    month_interval_each_year = get_month_intervals()
    # Segment the series timeseries into months from 1900 to 2100
    series_month_each = np.empty(
        (12, len(month_interval_each_year), len(timescale)), dtype=IndexedSnapshot
    )  # (month, year, scale)
    for i, year in enumerate(month_interval_each_year):
        for j, month in enumerate(year):
            for k, scale in enumerate(timescale):
                series_month_each[j, i, k] = series[month[0] : month[1]].rescale(scale)

    # IndexedSnapshot that stores the total of each month
    series_month_total = np.empty(
        (12, len(timescale)), dtype=IndexedSnapshot
    )  # (month, scale)
    for i in range(12):
        for j in range(len(series_month_each[0])):
            for k, scale in enumerate(timescale):
                if j == 0:
                    series_month_total[i, k] = IndexedSnapshot()
                series_month_total[i, k].add(
                    series_month_each[i, j, k], sequential=True
                )

    stats_month = np.zeros((12, len(timescale), 5))  # (month, scale, stats)
    for month in range(12):
        for scale in range(len(timescale)):
            model = series_month_total[month, scale]
            stats_month[month, scale, :] = [
                model.mean(),
                model.coef_variation(),
                model.autocorr_coef(1),
                model.skewness(),
                model.pDry(0),
            ]

    stats_month_seperate = np.zeros(
        (12, len(month_interval_each_year), len(timescale), 5)
    )  # (month, year, scale, stats)
    for month in range(12):
        for year in range(len(month_interval_each_year)):
            for scale in range(len(timescale)):
                model = series_month_each[month, year, scale]
                stats_month_seperate[month, year, scale, :] = [
                    model.mean(),
                    model.coef_variation(),
                    model.autocorr_coef(1),
                    model.skewness(),
                    model.pDry(0),
                ]

    stats_weight = 1 / np.nanvar(stats_month_seperate, axis=1)  # (month, scale, stats)

    return stats_month, stats_weight


def classic_statistics(
    timeseries: pd.Series[float], timescale: list[timedelta], intensity_time_interval: timedelta
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Calculate statistics of the input timeseries for model fitting.

    Parameters
    ----------
    timeseries : pd.Series
        Input timeseries with a DatetimeIndex or PeriodIndex and intensity
        values in mm/h.
    timescale : list of timedelta
        List of timescales to compute statistics over.
    intensity_time_interval : timedelta
        Time resolution of the input timeseries.

    Returns
    -------
    target_df_list : list of pd.DataFrame
        List of 12 DataFrames (one per month) with statistical targets.
        Columns are `StatMetrics`; index is timescale in hours.
    weight_df_list : list of pd.DataFrame
        List of 12 DataFrames (one per month) with fitting weights.
        Same structure as ``target_df_list``.
    """

    scale_second = np.array([ts.total_seconds() for ts in timescale], dtype=np.float64)
    scale_hr = scale_second / 3600

    # Copy the timeseries to avoid changing the original one
    timeseries = timeseries.copy()

    # Check the index type of the timeseries. It can be either DatetimeIndex or PeriodIndex
    if isinstance(timeseries.index, pd.DatetimeIndex):
        pass
    elif isinstance(timeseries.index, pd.PeriodIndex):
        timeseries.index = timeseries.index.to_timestamp()
    else:
        raise ValueError(f"The index of the timeseries should be either DatetimeIndex or PeriodIndex. Got {type(timeseries.index)}")

    # Check the type of the intensity data. It should be a number that can be converted to float
    if not pd.api.types.is_numeric_dtype(timeseries):
        raise ValueError(f"The intensity data should be numeric. Got {timeseries.dtype}")


    unix_time = timeseries.index.to_numpy().astype(np.float64) // 10**9
    intensity = timeseries.to_numpy()
    # Replace all the negative intensity to np.nan
    intensity[intensity < 0] = np.nan
    intensity = intensity / intensity_time_interval.total_seconds()

    target, weight = calculate_stats_matrix(unix_time, intensity, scale_second)

    target_df_list = []
    weight_df_list = []
    for month_idx in range(12):
        # Remove pDry from the target and weight
        target_df = pd.DataFrame(
            target[month_idx, :, :-1], columns=[StatMetrics.MEAN, StatMetrics.CVAR, StatMetrics.AR1, StatMetrics.SKEWNESS], index=scale_hr
        )
        weight_df = pd.DataFrame(
            weight[month_idx, :, :-1], columns=[StatMetrics.MEAN, StatMetrics.CVAR, StatMetrics.AR1, StatMetrics.SKEWNESS], index=scale_hr
        )
        target_df.index.name = "timescale_hr"
        weight_df.index.name = "timescale_hr"

        target_df_list.append(target_df)
        weight_df_list.append(weight_df)



    return target_df_list, weight_df_list
