import pandas as pd

from .utils import *


def createStatisticalFile(
    rawData,
    propertyList,
    timeScaleList,
    meanTimeScale,
    outputMean,
    outputPath,
    weightFile_path,
    pDryThreshold=0,
):
    month = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    rf_ts = {}
    data = {}
    weight = {}
    data["Month"] = month
    weight["Month"] = month
    start_time = rawData.index[0]
    end_time = rawData.index[-1]
    print(start_time, end_time)

    # step 4 calculate mean
    for sinmeanTimeScale in meanTimeScale:
        subSeq = change_timescale(rawData, sinmeanTimeScale)
        w_subSeq = change_timescale(rawData, "1h")
        col_data = []
        col_weight = []
        for m in range(1, 13):
            monthlyData = subSeq[subSeq.index.month == m]
            # print(f'============================== MONTH:{m} ===============================')
            # print(f'monthlyData: {monthlyData}')
            rf_ts[m] = monthlyData
            w_monthlyData = w_subSeq[w_subSeq.index.month == m]
            mData = monthlyData.mean()
            mWeight = calculateWeight(w_monthlyData, "Mean")
            col_data.append(round(mData, 7))
            col_weight.append(round(mWeight, 7))
        if sinmeanTimeScale == outputMean:
            data["{}_{}".format("Mean", str(sinmeanTimeScale))] = col_data
        else:
            continue
        weight["Mean_60"] = col_weight

    # step 5 calculate other properties
    for sinTimescale in timeScaleList:
        scaledData = change_timescale(rawData, sinTimescale)
        for sinStaProp in propertyList:
            col_data = []
            col_weight = []
            for curMonth in range(1, 13):
                monthlyData = scaledData[scaledData.index.month == curMonth]
                mWeight = calculateWeight(monthlyData, sinStaProp)
                # print(f'mWeight : {mWeight}')
                input = monthlyData.to_numpy().flatten()
                if sinStaProp == "CV":
                    ret = cal_CVAR(input)
                elif sinStaProp == "AR-1":
                    ret = cal_AR1(input)
                elif sinStaProp == "Skewness":
                    ret = cal_skewness(input)
                elif sinStaProp == "pDry":
                    ret = cal_pDry(input, pDryThreshold)
                col_data.append(round(ret, 7))
                # print(f'curMonth : {curMonth}')
                # print(f'col_data : {col_data}')
                col_weight.append(round(mWeight, 7))
                # print(f'col_weight : {col_weight}')
            data["{}_{}".format(str(sinStaProp), str(sinTimescale))] = col_data
            weight["{}_{}".format(str(sinStaProp), str(sinTimescale))] = col_weight

    # step 6 output the statistical properties and their weights
    pd.DataFrame(data).set_index("Month").to_csv(outputPath)
    pd.DataFrame(weight).set_index("Month").to_csv(weightFile_path)
    print("Statistical properties generated")

    # return rf_ts


def calculateWeight(monthlyData, sinStaProp, pDryThreshold=0):
    years = monthlyData.index.year.unique()
    if sinStaProp == "Mean":
        func = cal_mean
    elif sinStaProp == "CV":
        func = cal_CVAR
    elif sinStaProp == "AR-1":
        func = cal_AR1
    elif sinStaProp == "Skewness":
        func = cal_skewness
    elif sinStaProp == "pDry":
        func = cal_pDry

    # Check if monthlyData is Jan
    if monthlyData.index.month.unique() == 1 and sinStaProp == "AR-1":
        print("Monthly data is Jan")
        print(monthlyData[monthlyData.index.year == 1987].values)
        print(cal_AR1(monthlyData[monthlyData.index.year == 1987]))

    if sinStaProp != "pDry":
        stat_list = [
            func(monthlyData[monthlyData.index.year == curYear]) for curYear in years
        ]
    else:
        stat_list = [
            func(
                monthlyData[monthlyData.index.year == curYear], threshold=pDryThreshold
            )
            for curYear in years
        ]


    # return 1.0 / np.var(stat_list) if len(years) > 4 else (1 / np.mean(stat_list)) ** 2
    # print(f'Month : {monthlyData.index.month.unique()}, {stat_list}')
    return 1.0 / np.nanvar(stat_list)
