import numpy as np
import pandas as pd
import math

from datetime import datetime as dt
from datetime import timedelta as td


def MergeCells(storms, freq):
    units_in_second = {'5T': 300, '1H': 3600, '6H': 21600, '1D': 86400}
    unit = units_in_second[freq]

    # find boundaries of time interval
    min_time = np.min(np.array([storm.sDT for storm in storms]))
    max_time = np.max(np.array([cell.eDT for storm in storms for cell in storm.cells]))

    time_index = pd.date_range(min_time, max_time, normalize=False, freq=freq)
    rainfall_ts = pd.Series([0]*len(time_index), index=time_index)

    for storm in storms:
        for cell in storm.cells:
            # s, e - index of each cell's start time and end time
            s = math.floor((cell.sDT - min_time).total_seconds() / unit)
            e = math.floor((cell.eDT - min_time).total_seconds() / unit)

            cell_intensity = cell.Depth / 3600 * unit

            if e == s:
                rainfall_ts.loc[time_index[s]] = cell_intensity
                continue

            # calculate each time interval's rainfall portion
            head_depth = (time_index[s+1] - cell.sDT).total_seconds() / unit * cell_intensity
            tail_depth = (cell.eDT - time_index[e]).total_seconds() / unit * cell_intensity
            body_depth = cell_intensity

            # distribute rainfall to each time interval
            rainfall_ts.loc[time_index[s]] += head_depth
            rainfall_ts.loc[time_index[e]] += tail_depth
            rainfall_ts.loc[time_index[s+1]:time_index[e-1]] += body_depth

    return rainfall_ts