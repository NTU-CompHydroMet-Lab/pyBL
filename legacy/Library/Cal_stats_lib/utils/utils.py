import argparse

import numpy as np
import pandas as pd
import yaml
from scipy.stats import moment, skew, variation
from statsmodels.tsa.stattools import acovf


def change_timescale(target_dataframe, scale):
    return target_dataframe.resample(scale).sum()

# MEAN
def cal_mean(target_array):
    return np.mean(target_array)

# AR1
def cal_AR1(target_array):
    return pd.Series(target_array).autocorr(lag=1)

# SKEWNESS
# Scale is 5min, 1hr, 6hr, 24hr
def cal_skewness(target_array):
    return float(skew(target_array))

# Coefficient of variation. == CVAR
# Scale is 5min, 1hr, 6hr, 24hr
def cal_CVAR(target_array):
    return variation(target_array).item()
    # return float(variation(target_array, axis=0))

# pDRY
def cal_pDry(target_array, threshold=0):
    return 1 - np.count_nonzero(target_array > threshold) / len(target_array)
