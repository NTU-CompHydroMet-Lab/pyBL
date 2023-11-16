import os

import numpy as np
import pandas as pd

from pyBL.fitting.fitter import BLRPRxFitter
from pyBL.models import Stat_Props, BLRPRx, BLRPRx_params
from pyBL.timeseries import IntensityMRLE

from Library.Fitting_lib.objectiveFunction import Exponential_func
from Library.BLRPRmodel.BLRPRx import BLRPRx as BLRPRx_legacy

timescale = [1, 3600, 3*3600, 6*3600, 24*3600]
props = [Stat_Props.MEAN, Stat_Props.CVAR, Stat_Props.AR1, Stat_Props.SKEWNESS, Stat_Props.pDRY]

# Set timezone to UTC
os.environ['TZ'] = 'UTC'

def rain_timeseries():
    data_path = os.path.join(os.path.dirname(__file__), '01 Input_data', 'elmdon.csv')
    data = pd.read_csv(data_path, parse_dates=["datatime"])
    data['datatime'] = data['datatime'].astype("int64") // 10 ** 9
    time = data['datatime'].to_numpy()
    intensity = data['Elmdon'].to_numpy()
    return time, intensity

def month_start_end():
    # Generate first day of each month from 1980 to 2010
    day = pd.date_range(start='1980-01-01', end='2000-01-01', freq='MS')
    # Convert to unix time
    month_srt = day.astype("int64") // 10 ** 9
    month_end = month_srt
    # Stack them together
    month_interval = np.stack((month_srt[:-1], month_end[1:]), axis=1)
    # Group the month_interval by month
    month_interval = np.reshape(month_interval, (-1, 12, 2))
    return month_interval

time, intensity = rain_timeseries()
mrle = IntensityMRLE(time, intensity/3600)
month_interval_each_year = month_start_end()

# Segment the mrle timeseries into months from 1900 to 2100
mrle_month_each = np.empty((12, len(month_interval_each_year), 5), dtype=IntensityMRLE)    # (month, year, scale)
for i, year in enumerate(month_interval_each_year):
    for j, month in enumerate(year):
        for k, scale in enumerate(timescale):
            mrle_month_each[j, i, k] = mrle[month[0]:month[1]].rescale(scale)                   # 1s scale

# MRLE that stores the total of each month
mrle_month_total = np.empty((12, 5), dtype=IntensityMRLE)    # (month, scale)
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
        stats_month[month, scale, :] = [model.mean(), model.cvar(), model.acf(), model.skewness(), model.pDry(0)]

stats_month_seperate = np.zeros((12, len(month_interval_each_year), 5, 5))  # (month, year, scale, stats)
for month in range(12):
    for year in range(len(month_interval_each_year)):
        for scale in range(5):
            model = mrle_month_each[month, year, scale]
            stats_month_seperate[month, year, scale, :] = [model.mean(), model.cvar(), model.acf(), model.skewness(), model.pDry(0)]

stats_weight = 1/np.nanvar(stats_month_seperate, axis=1)  # (month, scale, stats) (12, 5, 5)

target = stats_month[:, 1:, :]
weight = stats_weight[:, 1:, :]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

params = BLRPRx_params(lambda_=0.016679733103341976, 
                       phi=0.08270236178820184, 
                       kappa=0.34970877070925505, 
                       alpha=9.017352714561754, 
                       nu=0.9931496975448589, 
                       sigmax_mux=1.0, 
                       iota=0.971862948182735)
legacy_params = [params.lambda_, params.iota, params.sigmax_mux, 0, params.alpha, params.alpha/params.nu, params.kappa, params.phi, 1.0]
fitter = BLRPRxFitter()
model = BLRPRx(params=params)

new_score = fitter.evaluate(x=model.get_params(), model=model, target=target[0], weight=weight[0])
legacy_target = [target[0, 0, 0], target[0, 0, 1], target[0, 0, 2], target[0, 0, 2], 
                target[0, 1, 1], target[0, 1, 3], target[0, 1, 2],
                target[0, 2, 1], target[0, 2, 3], target[0, 2, 2], 
                target[0, 3, 1], target[0, 3, 3], target[0, 3, 2]]
legacy_weight = [weight[0, 0, 0], weight[0, 0, 1], weight[0, 0, 3], weight[0, 0, 2], 
                weight[0, 1, 1], weight[0, 1, 3], weight[0, 1, 2],
                weight[0, 2, 1], weight[0, 2, 3], weight[0, 2, 2], 
                weight[0, 3, 1], weight[0, 3, 3], weight[0, 3, 2]]

legacy_target_cor = [target[0, 0, 0], target[0, 0, 1], target[0, 0, 2], target[0, 0, 3], 
                target[0, 1, 1], target[0, 1, 2], target[0, 1, 3],
                target[0, 2, 1], target[0, 2, 2], target[0, 2, 3], 
                target[0, 3, 1], target[0, 3, 2], target[0, 3, 3]]
legacy_weight_cor = [weight[0, 0, 0], weight[0, 0, 1], weight[0, 0, 2], weight[0, 0, 3], 
                weight[0, 1, 1], weight[0, 1, 2], weight[0, 1, 3],
                weight[0, 2, 1], weight[0, 2, 2], weight[0, 2, 3], 
                weight[0, 3, 1], weight[0, 3, 2], weight[0, 3, 3]]
old_score = Exponential_func(legacy_params, 0, [1, 3, 6, 24], legacy_target, legacy_weight, model=BLRPRx_legacy(mode='exponential'))
cor_score = Exponential_func(legacy_params, 0, [1, 3, 6, 24], legacy_target_cor, legacy_weight_cor, model=BLRPRx_legacy(mode='exponential'))
print(new_score, old_score, cor_score)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')