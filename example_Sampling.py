from dotmap import DotMap
from model.BLRPRx import *
from calendar import month_abbr
from datetime import timedelta as td
from datetime import datetime as dt
from datetime import datetime
from utils.utils import *
from utils.stats_calculation import *
import numpy as np
import pandas as pd
import os
from sampling.mergeCells import *
from sampling.sampling import *
from fitting import fitting, objectiveFunction
import warnings, yaml

args = yaml.load(open('./config/default.yaml'), Loader=yaml.FullLoader)
args = DotMap(args)
args.sampling.start_time = dt.strptime(args.sampling.start_time, '%Y-%m-%d')
args.sampling.end_time = dt.strptime(args.sampling.end_time, '%Y-%m-%d')

total_sample_sta_prop = []
num_month = pd.read_csv(args.IO.stats_file_path,index_col=0, header=0).shape[0]

# for month in range(1, num_month + 1):
for month in range(1, 4):

    start_time, end_time, freq = args.sampling.start_time, args.sampling.end_time, args.sampling.freq
    duration_sim_hr = (end_time - start_time).total_seconds() / 3600

    # step 0 read result thetas(lambda, iota, alpha, nu, kappa, phi)
    thetas = pd.read_csv('./Theta.csv').loc[month]

    # step 1 sample storms
    storms = SampleStorm(thetas, start_time, duration_sim_hr)
    # import pdb; pdb.set_trace()
    rainfall_ts = MergeCells(storms, freq)

    # step 2 calculate the sta prop of the sample ts
    sample_sta_prop = cal_sta_prop(rainfall_ts)
    print('The statistical prop of the sample rainfall data is {}'.format(["%.7f" % elem for elem in sample_sta_prop]))
    total_sample_sta_prop.append(sample_sta_prop)

    # if not os.path.exists('./result'):
    #     os.mkdir('./result')
    # rainfall_ts.to_csv('./result/{}_{}_{}.csv'.format(month_abbr[month], freq, (end_time-start_time).days//365))
    print("----------Finish row {} sampling!----------\n\n".format(month))

# step 3 output the sta prop
total_sample_sta_prop = np.array(total_sample_sta_prop)

titles = ['Mean 1H', 'CVAR 5Min', 'AR1 5Min', 'Skewness 5Min',
                     'CVAR 1H', 'AR1 1H', 'Skewness 1H',
                     'CVAR 6H', 'AR1 6H', 'Skewness 6H',
                     'CVAR 24H', 'AR1 24H', 'Skewness 24H']

if args.IO.store_sample_stats:
    pd.DataFrame(data=total_sample_sta_prop, columns=titles).to_csv(args.IO.sample_stats_path)
