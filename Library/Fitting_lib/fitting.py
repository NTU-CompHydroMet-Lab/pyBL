from scipy.optimize import dual_annealing, minimize, shgo, basinhopping, differential_evolution
import random
import numpy as np
import pandas as pd
import math
import scipy.special as sc
from Library.BLRPRmodel.BLRPRx import *

np.random.seed(1340)

def scalesTransform(timeScaleList):
    b = []
    for i in timeScaleList:
        if i[1] == 'a':
            a = np.nan
            continue
        elif i[1] == 'T':
            a = float(i[0])/60
        elif i[1] == 'h':
            a = int(i[0])
        elif i[1] == 'D':
            a = int(i[0])*24
        elif i[1] == 'M':
            a = int(i[0])*30*24
        elif i[1] == 'Y':
            a = int(i[0])*365*24
        else: a = i[0]
        b.append(float(a))
    return b

def drop_prop(df, month, timeScaleFile, Pdrop):
    timeScaleList = timeScaleFile.loc['time'].dropna().to_numpy()
    p = timeScaleFile.loc['prop'].dropna().tolist()
    if Pdrop in p:
        dr = [str(Pdrop) + '_' + i for i in timeScaleList]
        data = df.iloc[month]
        data_fin = data.drop(dr)
    else: print(f'There is no property {Pdrop} in the dataset! Please choose one from the property list.')
    return data_fin.to_numpy()


def Annealing(theta, obj_func, fitting_model, month, timeScaleList, staFile_path, weightFile_path, timescaleFile_path):
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
    data = drop_prop(df, month-1, timescaleFile, 'pDry')
    weight = drop_prop(Wdf, month-1, timescaleFile, 'pDry')
#     print(f'TimeScaleList : {timeScaleList}')
#     print(f'Theta : {theta}')
#     print(f'Data : {data}')
#     print(f'Weight : {weight}')
    
    ret = dual_annealing(obj_func, bounds=list(zip(lw, up)), x0=theta, maxiter=5000,
                         args=[month-1, timeScaleList, data, weight, fitting_model])

    theta = ret.x
    #print("The fval of Annealing = {}".format(ret.fun))
    return theta, ret.fun


def Differential_evolution(theta, obj_func, fitting_model, month, timeScaleList, staFile_path, weightFile_path):
    r"""
    Optimize the theta, the parameter of BL model by differential evolution which is a global optimization method provided by scipy
    :param theta: list, a list of parameters
    :param obj_func: function method, the model equation
    :param month: int, the index to assign the stats properties from stats file and weight file
    :param timeScaleList: list, a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    :param staFile_path: string, the file path of the statistical properties
    :param weightFile_path: string, the file path of the weight(uncertainty) of the statistical properties
    :return theta: list, the optimized theta parameters
    """
    if obj_func == None:
        return -9999

    n_theta = len(theta)

    xp0 = [item for item in theta if item >= 0]
    r = random.random()

    lw = [1e-6] * n_theta
    up = [20] * n_theta

    df = pd.read_csv(staFile_path, index_col=0, header=0)
    Wdf = pd.read_csv(weightFile_path, index_col=0, header=0)
    timescaleFile = pd.read_csv(timescaleFile_path, index_col=0, header=None)
    data = drop_prop(df, month-1, timescaleFile, 'pDry')
    weight = drop_prop(Wdf, month-1, timescaleFile, 'pDry')
    
    ret = differential_evolution(obj_func,  bounds=list(zip(lw, up)), args=(month-1, timeScaleList, data, weight, fitting_model),
                                 strategy='best1bin', maxiter=5000, disp=True, tol=1e-6)

    theta = ret.x
    
    print("The fval of Differential_evolution = {}".format(ret.fun))
    return theta


def Basinhopping(theta, obj_func, fitting_model, month, timeScaleList, staFile_path, weightFile_path, timescaleFile_path):
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
    data = drop_prop(df, month-1, timescaleFile, 'pDry')
    weight = drop_prop(Wdf, month-1, timescaleFile, 'pDry')
    
    ret = basinhopping(obj_func, theta,  niter=250,
                       minimizer_kwargs={'args': (month-1, timeScaleList, data, weight, fitting_model),
                                         'bounds': list(zip(lw, up))})
    theta = ret.x
    print("The fval of Basinhopping = {}".format(ret.fun))
    return theta, ret.fun


def Shgo(theta, obj_func, month, timeScaleList, staFile_path, weightFile_path):
    r"""
    Adopt shgo, a global optimization method for optimize the theta of the BL model.
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

    xp0 = [item for item in theta if item >= 0]
    r = random.random()

    lw = [1e-6] * n_theta
    up = [20] * n_theta

    df = pd.read_csv(staFile_path, index_col=0, header=0)
    Wdf = pd.read_csv(weightFile_path, index_col=0, header=0)
    timescaleFile = pd.read_csv(timescaleFile_path, index_col=0, header=None)
    data = drop_prop(df, month-1, timescaleFile, 'pDry')
    weight = drop_prop(Wdf, month-1, timescaleFile, 'pDry')

    ret = shgo(obj_func, bounds=list(zip(lw, up)), iters=3,
               args=[month-1, timeScaleList, data, weight])
    theta = ret.x
    # print(ret.message)
    print("The fval of shgo = {}".format(ret.fun))

    return theta


def Nelder_Mead(theta, obj_func, fitting_model, month, timeScaleList, staFile_path, weightFile_path, timescaleFile_path):
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
    data = drop_prop(df, month-1, timescaleFile, 'pDry')
    weight = drop_prop(Wdf, month-1, timescaleFile, 'pDry')

    method_list = ['L-BFGS-B', 'TNC', 'SLSQP',
                   'Powell', 'trust-constr']  # 'Nelder-Mead',
    score_list = []
    result_list = []
    for method in method_list:
        res = minimize(obj_func, theta, method=method,  bounds=list(zip(lw, up)),
                       args=(month-1, timeScaleList, data, weight, fitting_model), 
                       options={'disp': False, 'maxiter': 5000})

        score_list.append(res.fun)
        result_list.append(res.x)

    min_index = score_list.index(min(score_list))
    print('The best method is {}'.format(method_list[min_index]))
    print("The fval of local minimum = {}".format(score_list[min_index]))
    return result_list[min_index], score_list[min_index]

# def PSO( theta, obj_func, month, timeScaleList):
#     if obj_func == None:
#         return -9999
#     n_particles = 1000
#     n_theta = len(theta)
#     lw = np.zeros(9)
#     up = 20 * np.ones(9)
#     bounds = (lw, up)
#     xp0 =  np.array([[item for item in theta if item >= 0]])
#     init_pos = np.zeros((n_particles, xp0.shape[1]))
#     for row in range(n_particles):
#         init_pos[row, :] = xp0 + np.random.rand(1,9)
#     #global search part
#     # options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
#     # optimizer = ps.single.GlobalBestPSO(
#     #     n_particles=n_particles,
#     #     dimensions=9,
#     #     init_pos = init_pos,
#     #     options=options,
#     #     bounds=bounds
#     # )
#     # cost, global_res = optimizer.optimize(obj_func, iters=10000, month = month-1 )
#     # #transfer the global position to the start point of the local search
#     # global_pos = np.zeros((n_particles, xp0.shape[1]))
#     # for row in range(n_particles):
#     #     init_pos[row, :] = global_res + np.random.rand(1,9)
#     #local search part
#     options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 3, 'p': 2}
#     optimizer = ps.single.LocalBestPSO(n_particles=n_particles, dimensions=9,
#         init_pos = init_pos, options=options, bounds=bounds)
#     cost, local_pos = optimizer.optimize(obj_func, 10000, month = month-1, timeScaleList = timeScaleList)
#     return local_pos
