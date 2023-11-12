from pyBL.timeseries import IntensityMRLE
from pyBL.fitting import BLRPRxFitter
from pyBL.models import BLRPRx, Stat_Props, BLRPRx_params
import os
import pandas as pd
import numpy as np

timescale = [1, 3600, 3 * 3600, 6 * 3600, 24 * 3600]
props = [
    Stat_Props.MEAN,
    Stat_Props.CVAR,
    Stat_Props.AR1,
    Stat_Props.SKEWNESS,
    Stat_Props.pDRY,
]

# Set timezone to UTC
os.environ["TZ"] = "UTC"


def rain_timeseries():
    data_path = os.path.join(os.path.dirname(__file__), "data", "elmdon.csv")
    data = pd.read_csv(data_path, parse_dates=["datatime"])
    data["datatime"] = data["datatime"].astype("int64") // 10**9
    time = data["datatime"].to_numpy()
    intensity = data["Elmdon"].to_numpy()
    return time, intensity


def month_start_end():
    # Generate first day of each month from 1980 to 2010
    day = pd.date_range(start="1980-01-01", end="2000-01-01", freq="MS")
    # Convert to unix time
    month_srt = day.astype("int64") // 10**9
    month_end = month_srt
    # Stack them together
    month_interval = np.stack((month_srt[:-1], month_end[1:]), axis=1)
    # Group the month_interval by month
    month_interval = np.reshape(month_interval, (-1, 12, 2))
    return month_interval


time, intensity = rain_timeseries()
mrle = IntensityMRLE(time, intensity / 3600)
month_interval_each_year = month_start_end()

# Segment the mrle timeseries into months from 1900 to 2100
mrle_month_each = np.empty(
    (12, len(month_interval_each_year), 5), dtype=IntensityMRLE
)  # (month, year, scale)
for i, year in enumerate(month_interval_each_year):
    for j, month in enumerate(year):
        for k, scale in enumerate(timescale):
            mrle_month_each[j, i, k] = mrle[month[0] : month[1]].rescale(
                scale
            )  # 1s scale

# MRLE that stores the total of each month
mrle_month_total = np.empty((12, 5), dtype=IntensityMRLE)  # (month, scale)
for i in range(12):
    for j in range(len(mrle_month_each[0])):
        for k, scale in enumerate(timescale):
            if j == 0:
                mrle_month_total[i, k] = IntensityMRLE(scale=scale)
            mrle_month_total[i, k].add(mrle_month_each[i, j, k], sequential=True)

stats_month = np.zeros((12, 5, 5))  # (month, scale, stats)
for month in range(12):
    for scale in range(5):
        model = mrle_month_total[month, scale]
        stats_month[month, scale, :] = [
            model.mean(),
            model.cvar(),
            model.acf(),
            model.skewness(),
            model.pDry(0),
        ]

stats_month_seperate = np.zeros(
    (12, len(month_interval_each_year), 5, 5)
)  # (month, year, scale, stats)
for month in range(12):
    for year in range(len(month_interval_each_year)):
        for scale in range(5):
            model = mrle_month_each[month, year, scale]
            stats_month_seperate[month, year, scale, :] = [
                model.mean(),
                model.cvar(),
                model.acf(),
                model.skewness(),
                model.pDry(0),
            ]

stats_weight = 1 / np.nanvar(
    stats_month_seperate, axis=1
)  # (month, scale, stats) (12, 5, 5)

target = stats_month[:, 1:, :]
weight = stats_weight[:, 1:, :]

fitter = BLRPRxFitter()
result, model = fitter.fit(target[0], weight[0])
print(result)
print(result.x.tolist())
