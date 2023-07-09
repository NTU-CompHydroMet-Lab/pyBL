from dotmap import DotMap
from Library.BLRPRmodel.BLRPRx import *
from calendar import month_abbr
from datetime import timedelta as td
from datetime import datetime as dt
from datetime import datetime
from Library.Cal_stats_lib.utils.utils import *
from Library.Cal_stats_lib.utils.stats_calculation import *
import numpy as np
import pandas as pd
import os

from Library.Sampling_lib.mergeCells import *
from Library.Sampling_lib.sampling import *
from Library.Sampling_lib.compare import *
from Library.Fitting_lib import fitting, objectiveFunction
import warnings, yaml

args = yaml.load(open('./config/default.yaml'), Loader=yaml.FullLoader)
args = DotMap(args)
args.sampling.start_time = dt.strptime(args.sampling.start_time, '%Y-%m-%d')
args.sampling.end_time = dt.strptime(args.sampling.end_time, '%Y-%m-%d')

total_sample_sta_prop = []
num_month = pd.read_csv(args.IO.stats_file_path,index_col=0, header=0).shape[0]
pDryThreshold = 0.5
timeseries = []

for month in range(1, num_month+1):
    start_time, end_time, freq = args.sampling.start_time, args.sampling.end_time, args.sampling.freq
    duration_sim_hr = (end_time - start_time).total_seconds() / 3600

    # step 0 read result thetas(lambda, iota, alpha, nu, kappa, phi)
    thetas = pd.read_csv('./02 Output_data/Theta.csv').loc[month-1]
    print('='*50)
    print(f'Mon : {str(month)}/{str(num_month)}')

    # step 1 sample storms
    alpha, nu, phi,storms = SampleStorm(thetas, start_time, duration_sim_hr)

    #import pdb; pdb.set_trace()
    rainfall_ts = MergeCells(storms, freq)
    
    # step 2 calculate the sta prop of the sample ts
    sample_sta_prop = sample_Sta_prop(rainfall_ts, args.IO.config_path, pDryThreshold)
    total_sample_sta_prop.append(sample_sta_prop)
    print("----------Finish row {} sampling!----------\n\n".format(month))

#  step 3 output the sta prop
df = pd.DataFrame(total_sample_sta_prop[0])
for i in range(1, len(total_sample_sta_prop)):
    df2 = pd.DataFrame(total_sample_sta_prop[i])
    fin_df = pd.concat([df, df2])
    df = fin_df
fin_df.to_csv(args.IO.sample_stats_path)
# %%



