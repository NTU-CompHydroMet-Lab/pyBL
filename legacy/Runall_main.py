# %%
# 5T 1H 6H 1D
# 1H 3H 6H 1D
# datatime not datetime
from datetime import datetime as dt

import dotmap
from Library.Cal_stats_lib.utils.stats_calculation import *
from Library.Cal_stats_lib.utils.utils import *

args = yaml.load(open("./config/default.yaml"), Loader=yaml.FullLoader)
args = dotmap.DotMap(args)

# step 1 get the input
rawData = pd.read_csv("./01 Input_data/test.csv", index_col="datatime")
rawData.index = pd.to_datetime(rawData.index)
pDryThreshold = 0.5

# step 1.5 check the value
if len(rawData.columns.values) != 1:
    print("the raw data have more than one column!! Check again!!")
else:
    rawData = rawData[rawData.columns.values[0]]
    ## delete all the negative number
    ## assign flags
    rawData = rawData.drop(rawData.index[rawData < 0.0])

# step 2 read which kind of property
prop = pd.read_csv(args.IO.config_path, index_col=0, header=None)
propertyList = prop.loc["prop"].dropna().to_numpy()
print("The used properties: {}".format(propertyList))

# step 3 read the timescale
timeScaleList = prop.loc["time"].dropna().to_numpy()
meanTimeScale = prop.loc["timeForMean"].dropna().to_numpy()
outputMean = prop.loc["OutputTimeScaleForMean"].dropna().to_numpy()
print("The time scales: {}".format(timeScaleList))
print("The time scale of Mean: {}".format(meanTimeScale))
print(f"Output time scale for Mean: {outputMean}")

if outputMean[0] not in meanTimeScale.tolist():
    print("-" * 50)
    print(
        f"Mean will not be calculated because {outputMean} is not in timeForMean, please choose the scale from timeForMean."
    )

# step 4-6 calculate and output statistical properties and their weights
createStatisticalFile(
    rawData,
    propertyList,
    timeScaleList,
    meanTimeScale,
    outputMean,
    args.IO.stats_file_path,
    args.IO.weight_file_path,
    pDryThreshold,
)


from datetime import datetime
from datetime import datetime as dt

import dotmap
import numpy as np

# %%
# %%
# 2:40 per month
import pandas as pd
from Library.BLRPRmodel.BLRPRx import *
from Library.Cal_stats_lib.utils.utils import *
from Library.Fitting_lib import fitting, objectiveFunction

args = yaml.load(open("./config/default.yaml"), Loader=yaml.FullLoader)
args = dotmap.DotMap(args)

# step 0 read statistical properties and corresponding weights
num_month = pd.read_csv(args.IO.stats_file_path, index_col=0, header=0).shape[0]
TSF = pd.read_csv(args.IO.config_path, index_col=0, header=None)
timeScale = TSF.loc["time"].dropna().tolist()
timeScaleList = fitting.scalesTransform(timeScale)
print("There are {} rows to fit".format(num_month))
file = pd.read_csv("./02 Output_data/result.csv")
result_csv = []
sav = {}


# for each calendar month
for month in range(1, num_month + 1):
    print(f"====================== Month : {month} =======================")
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
    # print(f'Original theta : {theta}')

    # step 2 initialize objective function
    obj = objectiveFunction.Exponential_func

    # step 3 initialize fitting model
    if args.fitting.moment == "BLRPRx":
        fitting_model = BLRPRx(args.fitting.intensity_mode)

    # step 4 find approximate global optimum using Dual Annealing algorithm.
    past = datetime.now()
    final, score = fitting.Annealing(
        theta,
        obj,
        fitting_model,
        month,
        timeScaleList,
        args.IO.stats_file_path,
        args.IO.weight_file_path,
        args.IO.config_path,
    )

    print(
        "Time cost == {}, Row {} theta after Annealing == {}".format(
            (datetime.now() - past), month, ["%.9f" % elem for elem in final]
        )
    )
    if score >= 1:
        past = datetime.now()
        final, score = fitting.Basinhopping(
            theta,
            obj,
            fitting_model,
            month,
            timeScaleList,
            args.IO.stats_file_path,
            args.IO.weight_file_path,
            args.IO.config_path,
        )
        print(
            "Time cost == {}, Row {} theta after Basinhopping == {}".format(
                (datetime.now() - past), month, ["%.9f" % elem for elem in final]
            )
        )

    # step 5 try to find a better local minimum using Nelder Mead algorithm
    past = datetime.now()
    local_final, local_score = fitting.Nelder_Mead(
        final,
        obj,
        fitting_model,
        month,
        timeScaleList,
        args.IO.stats_file_path,
        args.IO.weight_file_path,
        args.IO.config_path,
    )

    print(
        "Time cost == {}, Row {} theta after Nelder_Mead == {}".format(
            (datetime.now() - past), month, ["%.9f" % elem for elem in local_final]
        )
    )

    final = local_final if local_score < score else final
    final_stats = objectiveFunction.Cal(final, timeScaleList, fitting_model)
    sav[month] = final_stats
    print(f"Final Stats : {final_stats}")

    # do some minor modifications
    # delete useless variables
    final = np.delete(final, [2, 3])
    final[-1] = 1.0
    result_csv.append(final)
    file.loc[month] = final

prop = pd.read_csv(args.IO.config_path, index_col=0, header=None)
propertyList = prop.loc["prop"].dropna().to_numpy().tolist()

cols = []
for sinscale in timeScale:
    for sinStaprop in propertyList[:-1]:
        cols.append(str(sinStaprop) + "_" + str(sinscale))

cols.insert(0, "Mean_1h")
# step 6 output the result theta
file.to_csv("./02 Output_data/Theta.csv", index=False)

df = pd.DataFrame(sav).T
df.columns = cols
df.to_csv("Mfinal_staProp.csv")
print("Calibration finished")

# %%
# %%
# resample will compensate all empty timestemp
# 1h 3h 6h 24h if ih sample 5T

from datetime import datetime
from datetime import datetime as dt

import numpy as np
import pandas as pd
import yaml
from dotmap import DotMap
from Library.BLRPRmodel.BLRPRx import *
from Library.Cal_stats_lib.utils.stats_calculation import *
from Library.Cal_stats_lib.utils.utils import *
from Library.Fitting_lib import fitting, objectiveFunction
from Library.Sampling_lib.mergeCells import *
from Library.Sampling_lib.sampling import *

args = yaml.load(open("./config/default.yaml"), Loader=yaml.FullLoader)
args = DotMap(args)
args.sampling.start_time = dt.strptime(args.sampling.start_time, "%Y-%m-%d")
args.sampling.end_time = dt.strptime(args.sampling.end_time, "%Y-%m-%d")

total_sample_sta_prop = []
num_month = pd.read_csv(args.IO.stats_file_path, index_col=0, header=0).shape[0]
pDryThreshold = 0.5
timeseries = []

for month in range(1, num_month + 1):
    start_time, end_time, freq = (
        args.sampling.start_time,
        args.sampling.end_time,
        args.sampling.freq,
    )
    duration_sim_hr = (end_time - start_time).total_seconds() / 3600

    # step 0 read result thetas(lambda, iota, alpha, nu, kappa, phi)
    thetas = pd.read_csv("./02 Output_data/Theta.csv").loc[month - 1]
    print("=" * 50)
    print(f"Mon : {str(month)}/{str(num_month)}")

    # step 1 sample storms
    alpha, nu, phi, storms = SampleStorm(thetas, start_time, duration_sim_hr)

    # import pdb; pdb.set_trace()
    rainfall_ts = MergeCells(storms, freq)

    # step 2 calculate the sta prop of the sample ts
    sample_sta_prop = sample_Sta_prop(rainfall_ts, args.IO.config_path, pDryThreshold)
    total_sample_sta_prop.append(sample_sta_prop)
    print("----------Finish row {} sampling!----------\n\n".format(month))
    timeseries.append(rainfall_ts)

#  step 3 output the sta prop
df = pd.DataFrame(total_sample_sta_prop[0])
for i in range(1, len(total_sample_sta_prop)):
    df2 = pd.DataFrame(total_sample_sta_prop[i])
    fin_df = pd.concat([df, df2])
    df = fin_df

print("Sampling Finished")
fin_df.to_csv(args.IO.sample_stats_path)
print("All tasks are done")
# %%
