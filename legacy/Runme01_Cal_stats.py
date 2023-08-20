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


# %%
