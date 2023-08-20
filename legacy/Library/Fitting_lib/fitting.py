import math
import random

import numpy as np
import pandas as pd
import scipy.special as sc
from scipy.optimize import (basinhopping, differential_evolution,
                            dual_annealing, minimize, shgo)

from Library.BLRPRmodel.BLRPRx import *

np.random.seed(1340)


def scalesTransform(timeScaleList):
    b = []
    for i in timeScaleList:
        if i[1] == "a":
            a = np.nan
            continue
        elif i[1] == "T":
            a = float(i[0]) / 60
        elif i[1] == "h":
            a = int(i[0])
        elif i[1] == "D":
            a = int(i[0]) * 24
        elif i[1] == "M":
            a = int(i[0]) * 30 * 24
        elif i[1] == "Y":
            a = int(i[0]) * 365 * 24
        else:
            a = i[0]
        b.append(float(a))
    return b


def drop_prop(df, month, timeScaleFile, Pdrop):
    timeScaleList = timeScaleFile.loc["time"].dropna().to_numpy()
    p = timeScaleFile.loc["prop"].dropna().tolist()
    if Pdrop in p:
        dr = [str(Pdrop) + "_" + i for i in timeScaleList]
        data = df.iloc[month]
        data_fin = data.drop(dr)
    else:
        print(
            f"There is no property {Pdrop} in the dataset! Please choose one from the property list."
        )
    return data_fin.to_numpy()


def Annealing(
    theta,
    obj_func,
    fitting_model,
    month,
    timeScaleList,
    staFile_path,
    weightFile_path,
    timescaleFile_path,
):
    r"""
    Optimize the theta, the parameter of BL model by dual annealing which is a global optimization method provided by scipy
    :param theta: list, a list of parameters
    :param obj_func: function method, the model equation
    :param month: int, the index to assign the stats properties from stats file and weight file
    :param timeScaleList: list, a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    :param staFile_path: string, the file path of the statistical properties
    :param weightFile_path: string, the file path of the weight(uncertainty) of the statistical properties
    :return (theta, ret.fun): (list, float), the dict contains the optimized theta and the score for checking if the optimizatiom is correct
    """
    if obj_func == None:
        return -9999

    n_theta = len(theta)

    xp0 = [item for item in theta if item >= 0]
    r = random.random()

    lw = [1e-6] * n_theta
    up = [20] * n_theta
    # up[1] = 1.0

    df = pd.read_csv(staFile_path, index_col=0, header=0)
    Wdf = pd.read_csv(weightFile_path, index_col=0, header=0)
    timescaleFile = pd.read_csv(timescaleFile_path, index_col=0, header=None)
    data = drop_prop(df, month - 1, timescaleFile, "pDry")
    weight = drop_prop(Wdf, month - 1, timescaleFile, "pDry")
    #     print(f'TimeScaleList : {timeScaleList}')
    #     print(f'Theta : {theta}')
    #     print(f'Data : {data}')
    #     print(f'Weight : {weight}')

    ret = dual_annealing(
        obj_func,
        bounds=list(zip(lw, up)),
        x0=theta,
        maxiter=5000,
        args=[month - 1, timeScaleList, data, weight, fitting_model],
    )

    theta = ret.x
    # print("The fval of Annealing = {}".format(ret.fun))
    return theta, ret.fun


def Basinhopping(
    theta,
    obj_func,
    fitting_model,
    month,
    timeScaleList,
    staFile_path,
    weightFile_path,
    timescaleFile_path,
):
    r"""
    Optimize the theta, the parameter of BL model by basinhopping which is a global optimization method provided by scipy
    :param theta: list, a list of parameters
    :param obj_func: function method, the model equation
    :param month: int, the index to assign the stats properties from stats file and weight file
    :param timeScaleList: list, a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    :param staFile_path: string, the file path of the statistical properties
    :param weightFile_path: string, the file path of the weight(uncertainty) of the statistical properties
    :return (theta, ret.fun), (list, float), the dict contains the optimized theta and the score for checking if the optimizatiom is correct
    """
    if obj_func == None:
        return -9999

    n_theta = len(theta)

    xp0 = [item for item in theta if item >= 0]
    r = random.random()

    lw = [1e-6] * n_theta
    up = [20] * n_theta
    # up[1] = 1.0

    df = pd.read_csv(staFile_path, index_col=0, header=0)
    Wdf = pd.read_csv(weightFile_path, index_col=0, header=0)
    timescaleFile = pd.read_csv(timescaleFile_path, index_col=0, header=None)
    data = drop_prop(df, month - 1, timescaleFile, "pDry")
    weight = drop_prop(Wdf, month - 1, timescaleFile, "pDry")

    ret = basinhopping(
        obj_func,
        theta,
        niter=250,
        minimizer_kwargs={
            "args": (month - 1, timeScaleList, data, weight, fitting_model),
            "bounds": list(zip(lw, up)),
        },
    )
    theta = ret.x
    print("The fval of Basinhopping = {}".format(ret.fun))
    return theta, ret.fun


def Nelder_Mead(
    theta,
    obj_func,
    fitting_model,
    month,
    timeScaleList,
    staFile_path,
    weightFile_path,
    timescaleFile_path,
):
    r"""
    Optimize the theta, the parameter of BL model by local minimization optimization methods provided by scipy
    :param theta: list, a list of parameters
    :param obj_func: function method, the model equation
    :param month: int, the index to assign the stats properties from stats file and weight file
    :param timeScaleList: list, a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    :param staFile_path: string, the file path of the statistical properties
    :param weightFile_path: string, the file path of the weight(uncertainty) of the statistical properties
    :return theta: list, the optimized theta
    """
    if obj_func == None:
        return -9999

    n_theta = len(theta)
    lw = [1e-6] * n_theta
    up = [20] * n_theta
    # up[1] = 1.0

    df = pd.read_csv(staFile_path, index_col=0, header=0)
    Wdf = pd.read_csv(weightFile_path, index_col=0, header=0)

    timescaleFile = pd.read_csv(timescaleFile_path, index_col=0, header=None)
    data = drop_prop(df, month - 1, timescaleFile, "pDry")
    weight = drop_prop(Wdf, month - 1, timescaleFile, "pDry")

    method_list = [
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Powell",
        "trust-constr",
    ]  # 'Nelder-Mead',
    score_list = []
    result_list = []
    for method in method_list:
        res = minimize(
            obj_func,
            theta,
            method=method,
            bounds=list(zip(lw, up)),
            args=(month - 1, timeScaleList, data, weight, fitting_model),
            options={"disp": False, "maxiter": 5000},
        )

        score_list.append(res.fun)
        result_list.append(res.x)

    min_index = score_list.index(min(score_list))
    print("The best method is {}".format(method_list[min_index]))
    print("The fval of local minimum = {}".format(score_list[min_index]))
    return result_list[min_index], score_list[min_index]

