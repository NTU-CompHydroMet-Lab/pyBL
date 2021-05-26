import dotmap
from datetime import datetime as dt
from utils.utils import *

args = yaml.load(open('./config/default.yaml'), Loader=yaml.FullLoader)
args = dotmap.DotMap(args)

def createStatisticalFile(rawData, propertyList, timeScaleList, outputPath, weightFile_path):
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = {}
    weight = {}
    data['Month'] = month
    weight['Month'] = month

    # step 4 calculate mean
    subSeq = change_timescale(rawData, '1h')

    col_data = []
    col_weight = []
    for m in range(1, 13):
        monthlyData = subSeq[subSeq.index.month == m]
        mData = monthlyData.mean().item()
        mWeight = calculateWeight(monthlyData, 'Mean')
        col_data.append(round(mData, 7))
        col_weight.append(round(mWeight, 7))
    data['Mean_60'] = col_data
    weight['Mean_60'] = col_weight

    # step 5 calculate other properties
    for sinTimescale in timeScaleList:
        scaledData = change_timescale(rawData, sinTimescale)
        for sinStaProp in propertyList:
            col_data = []
            col_weight = []
            for curMonth in range(1, 13):
                monthlyData = scaledData[scaledData.index.month == curMonth]
                mWeight = calculateWeight(monthlyData, sinStaProp)
                input = monthlyData.to_numpy().flatten()
                if sinStaProp == 'CV':
                    ret = cal_CVAR(input)
                elif sinStaProp == 'AR-1':
                    ret = cal_AR1(input)
                elif sinStaProp == 'Skewness':
                    ret = cal_skewness(input)
                col_data.append(round(ret, 7))
                col_weight.append(round(mWeight, 7))
            data['{}_{}'.format(str(sinStaProp), str(sinTimescale))] = col_data
            weight['{}_{}'.format(str(sinStaProp), str(sinTimescale))] = col_weight

    # step 6 output the statistical properties and their weights
    pd.DataFrame(data).set_index('Month').to_csv(outputPath)
    pd.DataFrame(weight).set_index('Month').to_csv(weightFile_path)

def calculateWeight(monthlyData, sinStaProp):
    years = monthlyData.index.year.unique()
    if sinStaProp == 'Mean':
        func = cal_mean
    elif sinStaProp == 'CV':
        func = cal_CVAR
    elif sinStaProp == 'AR-1':
        func = cal_AR1
    elif sinStaProp == 'Skewness':
        func = cal_skewness
    stat_list = [func(monthlyData[monthlyData.index.year == curYear]) for curYear in years]
    # return 1.0 / np.var(stat_list) if len(years) > 4 else (1 / np.mean(stat_list)) ** 2
    return 1.0 / np.var(stat_list)

# step 1 get the input
rawData = pd.read_csv('./data/test.csv', index_col='datatime')
rawData.index = pd.to_datetime(rawData.index)

# step 1.5 check the value
if len(rawData.columns.values) != 1:
    print('the raw data have more than one column!! Check again!!')
else:
    rawData = rawData[rawData.columns.values[0]]
    ## delete all the negative number
    ## assign flags
    rawData = rawData.drop(rawData.index[rawData < 0.0])

# step 2 read which kind of property
prop = pd.read_csv(args.IO.config_path, index_col=0, header=None)
propertyList = prop.loc['prop'].dropna().to_numpy()
print('The used properties: {}'.format(propertyList))

# step 3 read the timescale
timeScaleList = prop.loc['time'].dropna().to_numpy()
print('The time scales: {}'.format(timeScaleList))

# step 4-6 calculate and output statistical properties and their weights
createStatisticalFile(rawData, propertyList, timeScaleList, args.IO.stats_file_path, args.IO.weight_file_path)