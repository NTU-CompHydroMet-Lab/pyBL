from .utils import *
import pandas as pd


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
    stat_list = [func(monthlyData[monthlyData.index.year == curYear])
                 for curYear in years]
    # return 1.0 / np.var(stat_list) if len(years) > 4 else (1 / np.mean(stat_list)) ** 2
    return 1.0 / np.var(stat_list)


def cal_sta_prop(sub_data):
    sub_data_1h = sub_data.resample('1H').sum()
    sub_data_6h = sub_data.resample('6H').sum()
    sub_data_24h = sub_data.resample('1D').sum()
    mean = cal_mean(sub_data_1h)
    mean = mean if type(mean) == float else mean.item()
    return_array = np.array([
        mean,
        cal_CVAR(sub_data),
        cal_AR1(sub_data.to_numpy().flatten()),
        cal_skewness(sub_data),
        cal_CVAR(sub_data_1h),
        cal_AR1(sub_data_1h.to_numpy().flatten()),
        cal_skewness(sub_data_1h),
        cal_CVAR(sub_data_6h),
        cal_AR1(sub_data_6h.to_numpy().flatten()),
        cal_skewness(sub_data_6h),
        cal_CVAR(sub_data_24h),
        cal_AR1(sub_data_24h.to_numpy().flatten()),
        cal_skewness(sub_data_24h), ])
    return return_array


def create_stats_file(rawData, propertyList, timeScaleList, outputPath, weightFile_path):
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data = {}
    weight = {}
    data['Month'] = month
    weight['Month'] = month
    # 先算mean
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
    # 算其他的
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
            weight['{}_{}'.format(
                str(sinStaProp), str(sinTimescale))] = col_weight
    df = pd.DataFrame(data).set_index('Month').to_csv(outputPath)
    df = pd.DataFrame(weight).set_index('Month').to_csv(weightFile_path)
