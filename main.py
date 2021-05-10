from dotmap import DotMap
from model.BLRPRx import *
from calendar import month_abbr
from datetime import timedelta as td
from datetime import datetime as dt
from datetime import datetime
from utils import *
from utils.stats_calculation import *
import numpy as np
import pandas as pd
import os
from sampling.mergeCells import *
from sampling.sampling import *
from fitting import fitting, objectiveFunction
import warnings, yaml
warnings.filterwarnings("ignore")

def main(args):
    # calculate the statistical property
    # step 1 get the input
    rawData = pd.read_csv(args.IO.raw_data_path, index_col='datatime')
    rawData.index = pd.to_datetime(rawData.index)
    # step 1.5 check the value
    if len(rawData.columns.values) != 1:
        print('the raw data have more than one column!! Check again!!')
    else:
        rawData = rawData[rawData.columns.values[0]]
        # delete all the negative number
        # assign flags
        rawData = rawData.drop(rawData.index[rawData < 0.0])

    # step 2 read which kind of property
    prop = pd.read_csv(args.IO.config_path, index_col=0, header=None)
    propertyList = prop.loc['prop'].dropna().to_numpy()
    print('The used properties: {}'.format(propertyList))

    # step 3 read the timescale
    timeScaleList = prop.loc['time'].dropna().to_numpy()
    print('The time scales: {}'.format(timeScaleList))
    # output the  statistical file
    create_stats_file(rawData, propertyList, timeScaleList,
                          args.IO.stats_file_path, args.IO.weight_file_path)

    # fitting
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    result_csv = []
    total_sample_sta_prop = []
    file = pd.read_csv('./result.csv')

    num_month = pd.read_csv(args.IO.stats_file_path,
                            index_col=0, header=0).shape[0]
    print('There are {} rows to fit'.format(num_month))
    for month in range(1, num_month + 1):
        # handling input
        theta = [None] * 9
        if args.fitting.theta_initial is None:
            theta[0] = 0.01  # lambda #
            theta[1] = 0.1  # iota #
            theta[2] = 1.0  # sigma/mu
            theta[3] = 1e-6  # spare space
            theta[4] = 2.1  # alpha #
            theta[5] = 7.1  # alpha/nu
            theta[6] = 0.1  # kappa #
            theta[7] = 0.1  # phi #
            theta[8] = 1.0  # c
        else:
            theta = args.fitting.theta_initial

        # initialize objective function
        obj = objectiveFunction.Exponential_func

        # initialize fitting model
        if args.fitting.moment == 'BLRPRx':
            fitting_model = BLRPRx(args.fitting.intensity_mode)

        # handling time range
        try:
            timerange = pd.read_csv(
                args.IO.config_path, index_col=0, header=None)
            timeScaleList = handleTimeRange(timerange.loc['time'].to_numpy())
        except:
            timeScaleList = [1/12, 1, 6, 24]

        # fitting

        # Annealing (test phase)
        past = datetime.now()
        final, score = fitting.Annealing(
            theta, obj, fitting_model, month, timeScaleList, args.IO.stats_file_path, args.IO.weight_file_path)
        print('Time cost == {}, Row {} theta after Annealing == {}'.format((datetime.now() - past), month,
                                                                           ["%.9f" % elem for elem in final]))

        if score >= 1:
            # Basinhopping
            # if score >= 1:
            past = datetime.now()
            final, score = fitting.Basinhopping(
                theta, obj, fitting_model, month, timeScaleList, args.IO.stats_file_path, args.IO.weight_file_path)
            print('Time cost == {}, Row {} theta after Basinhopping == {}'.format((datetime.now() - past), month,
                                                                                  ["%.9f" % elem for elem in final]))

        # Local Minimization
        # Nelder mead
        past = datetime.now()
        local_final, local_score = fitting.Nelder_Mead(
            final, obj, fitting_model, month, timeScaleList, args.IO.stats_file_path, args.IO.weight_file_path)
        print('Time cost == {}, Row {} theta after Nelder_Mead == {}'.format((datetime.now() - past), month,
                                                                             ["%.9f" % elem for elem in local_final]))

        final = local_final if local_score < score else final

        # do some miner modified
        # delete useless variables
        final = np.delete(final, [2, 3])

        final[-1] = 1.0
        result_csv.append(final)
        file.loc[month] = final

        # -------sampling stage---------

        start_time, end_time, freq = args.sampling.start_time, args.sampling.end_time, args.sampling.freq

        duration_sim_hr = (end_time - start_time).total_seconds() / 3600

        # thetas = [lambda, iota, alpha, nu, kappa, phi]
        thetas = final

        # start sampling
        storms = SampleStorm(thetas, start_time, duration_sim_hr)
        # import pdb; pdb.set_trace()
        rainfall_ts = MergeCells(storms, freq)

        # calculate the sta prop of the sample ts
        sample_sta_prop = cal_sta_prop(rainfall_ts)
        print('The statistical prop of the sample rainfall data is {}'.format(
            ["%.7f" % elem for elem in sample_sta_prop]))
        total_sample_sta_prop.append(sample_sta_prop)

        # if not os.path.exists('./result'):
        #     os.mkdir('./result')
        # rainfall_ts.to_csv('./result/{}_{}_{}.csv'.format(month_abbr[month], freq, (end_time-start_time).days//365))
        print("----------Finish row {} sampling!----------\n\n".format(month))
    file.to_csv(args.IO.theta_store_path, index=False)

    # store the sta prop
    total_sample_sta_prop = np.array(total_sample_sta_prop)

    titles = ['Mean 1H', 'CVAR 5Min', 'AR1 5Min', 'Skewness 5Min',
              'CVAR 1H', 'AR1 1H', 'Skewness 1H',
              'CVAR 6H', 'AR1 6H', 'Skewness 6H',
              'CVAR 24H', 'AR1 24H', 'Skewness 24H']

    if args.IO.store_sample_stats:
        pd.DataFrame(data=total_sample_sta_prop, columns=titles).to_csv(
            args.IO.sample_stats_path)


if __name__ == "__main__":
    args = yaml.load(open('./config/default.yaml'), Loader=yaml.FullLoader)
    args = DotMap(args)
    args.sampling.start_time = dt.strptime(
        args.sampling.start_time, '%Y-%m-%d')
    args.sampling.end_time = dt.strptime(args.sampling.end_time, '%Y-%m-%d')
    main(args)
