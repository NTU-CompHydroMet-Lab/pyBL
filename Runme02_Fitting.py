# %%
#2:40 per month
import pandas as pd
import argparse
import numpy as np
from Library.Cal_stats_lib.utils.utils import *
import dotmap
from datetime import datetime as dt
from Library.BLRPRmodel.BLRPRx import *

from Library.Fitting_lib import fitting, objectiveFunction
from datetime import datetime

args = yaml.load(open('./config/default.yaml'), Loader=yaml.FullLoader)
args = dotmap.DotMap(args)

# step 0 read statistical properties and corresponding weights
num_month = pd.read_csv(args.IO.stats_file_path, index_col=0, header=0).shape[0]
TSF = pd.read_csv(args.IO.config_path, index_col=0, header=None)
timeScale = TSF.loc['time'].dropna().tolist()
timeScaleList = fitting.scalesTransform(timeScale)
print('There are {} rows to fit'.format(num_month))
file = pd.read_csv('./02 Output_data/result.csv')
result_csv = []
sav = {}


# for each calendar month
for month in range(1,  num_month+1):
    print(f'====================== Month : {month} =======================')
    # step 1 set initial theta
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
    #print(f'Original theta : {theta}')

    # step 2 initialize objective function
    obj = objectiveFunction.Exponential_func

    # step 3 initialize fitting model
    if args.fitting.moment == 'BLRPRx':
        fitting_model = BLRPRx(args.fitting.intensity_mode)

    # step 4 find approximate global optimum using Dual Annealing algorithm.
    past = datetime.now()
    final, score = fitting.Annealing(
        theta, obj, fitting_model, month, timeScaleList, args.IO.stats_file_path, args.IO.weight_file_path, args.IO.config_path)
    
    print('Time cost == {}, Row {} theta after Annealing == {}'.format((datetime.now() - past), month,
                                                                       ["%.9f" % elem for elem in final]))
    if score >= 1:
        past = datetime.now()
        final, score = fitting.Basinhopping(theta, obj, fitting_model, month, timeScaleList,
                                            args.IO.stats_file_path, args.IO.weight_file_path, args.IO.config_path)
        print('Time cost == {}, Row {} theta after Basinhopping == {}'.format((datetime.now() - past),
                                                                              month,["%.9f" % elem for elem in final]))
    
    # step 5 try to find a better local minimum using Nelder Mead algorithm
    past = datetime.now()
    local_final, local_score = fitting.Nelder_Mead(
        final, obj, fitting_model, month, timeScaleList, args.IO.stats_file_path, args.IO.weight_file_path, args.IO.config_path)
    
    print('Time cost == {}, Row {} theta after Nelder_Mead == {}'.format((datetime.now() - past), month,
                                                                         ["%.9f" % elem for elem in local_final]))

    final = local_final if local_score < score else final
    final_stats = objectiveFunction.Cal(final, timeScaleList, fitting_model)
    sav[month] = final_stats
    print(f'Final Stats : {final_stats}')
    
    # do some minor modifications
    # delete useless variables
    final = np.delete(final, [2, 3])
    final[-1] = 1.0
    result_csv.append(final)
    file.loc[month] = final
    
prop = pd.read_csv(args.IO.config_path, index_col=0, header=None)
propertyList = prop.loc['prop'].dropna().to_numpy().tolist()
    
cols = []
for sinscale in timeScale:
    for sinStaprop in propertyList[:-1]:
        cols.append(str(sinStaprop) + '_' + str(sinscale))

cols.insert(0,'Mean_1h')
# step 6 output the result theta
file.to_csv('./02 Output_data/Theta.csv', index=False)

df = pd.DataFrame(sav).T
df.columns = cols
df.to_csv('Mfinal_staProp.csv')
print('Done')

# %%



