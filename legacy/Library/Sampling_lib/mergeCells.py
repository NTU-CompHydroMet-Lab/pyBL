from datetime import datetime as dt
from datetime import timedelta as td

import numpy as np
import pandas as pd
import yaml
from dotmap import DotMap


def MergeCells(storms, freq):
    min_time = np.min(np.array([storm.sDT for storm in storms])).replace(microsecond=0)
    max_time = np.max(
        np.array([cell.eDT for storm in storms for cell in storm.cells])
    ).replace(microsecond=0)
    args = yaml.load(open("./config/default.yaml"), Loader=yaml.FullLoader)
    args = DotMap(args)
    end_time = args.sampling.end_time
    max_time = np.max(
        np.array(
            [
                cell.eDT
                for storm in storms
                for cell in storm.cells
                if cell.eDT < dt.strptime(end_time, "%Y-%m-%d") + td(days=30)
            ]
        )
    ).replace(microsecond=0)

    whole_duration = max_time - min_time
    print(whole_duration)
    blanks = int(whole_duration.total_seconds())
    ts = np.zeros((blanks + 1))

    for storm in storms:
        for cell in storm.cells:
            s = int((cell.sDT.replace(microsecond=0) - min_time).total_seconds())
            e = int((cell.eDT.replace(microsecond=0) - min_time).total_seconds())
            real_depth_per_second = cell.Depth / 3600
            ts[s:e] += real_depth_per_second

    time_index = pd.date_range(min_time, max_time, freq="S")
    rainfall_ts = pd.Series(ts, index=time_index).resample(freq).sum()
    return rainfall_ts
