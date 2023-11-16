import numpy as np
import pandas as pd
from pyBL.timeseries import IntensityMRLE

def month_interval():
    # Generate first day of each month from 1980 to 2010
    day = pd.date_range(start='1900-01-01', end='2100-01-01', freq='MS')
    # Convert to unix time
    month_srt = day.astype("int64") // 10 ** 9
    month_end = month_srt
    # Stack them together
    month_interval = np.stack((month_srt[:-1], month_end[1:]), axis=1)
    # Group the month_interval by month
    month_interval = np.reshape(month_interval, (-1, 12, 2))
    return month_interval

