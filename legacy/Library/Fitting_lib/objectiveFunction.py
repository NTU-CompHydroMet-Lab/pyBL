import math

import numpy as np
import pandas as pd

from Library.BLRPRmodel.BLRPRx import *


def Exponential_func(x, month, timeScaleList, data, weight, model):
    # print("inital x =  {}".format(x))
    month = month
    M_state_h = Cal(x, timeScaleList, model)
    S = 0.0

    # Mean_60,CV_5,AR-1_5,Skewness_5,CV_60,AR-1_60,Skewness_60,CV_360,AR-1_360,Skewness_360,CV_1440,AR-1_1440,Skewness_1440
    for sin_x, compare_x, W in zip(M_state_h, data, weight):
        # print('sin_x = {}, compare_x = {}, W = {}'.format(sin_x, compare_x, W))
        dt = sin_x - compare_x
        S += W * dt * dt
    return S


def Cal(x, timeScaleList, model):
    _Props = ["CVAR", "AR1", "SKEWNESS"]
    M_state_h = []

    x[2] = 1.0  #  sigma/mu
    x[3] = 1e-6  #  spare space
    x[-1] = 1.0
    theta = np.insert(x, 0, 1)
    M_state_h.append(Get_Props("MEAN", theta, model))
    for p in timeScaleList:
        for prop in _Props:
            theta = np.insert(x, 0, p)
            M_state_h.append(Get_Props(prop, theta, model))
    return M_state_h


def Get_Props(Prop, theta, model):
    m_stat = -9999.0

    if Prop == "MEAN":
        m_stat = model.Mean(theta)
    elif Prop == "VAR":
        m_stat = model.Var(theta)
    elif Prop == "CVAR":
        m_stat = math.sqrt(model.Var(theta)) / model.Mean(theta)
    elif Prop == "AR1":
        m_stat = model.Cov(theta, 1) / model.Var(theta)
    elif Prop == "AR2":
        m_stat = model.Cov(theta, 2) / model.Var(theta)
    elif Prop == "AR3":
        m_stat = model.Cov(theta, 3) / model.Var(theta)
    elif Prop == "AC1":
        m_stat = model.Cov(theta, 1)
    elif Prop == "AC2":
        m_stat = model.Cov(theta, 2)
    elif Prop == "AC3":
        m_stat = model.Cov(theta, 3)
    elif Prop == "SKEWNESS":
        stdev3 = model.Var(theta) * math.sqrt(model.Var(theta))
        m_stat = model.Mom3(theta) / stdev3
    elif Prop == "pDRY":
        m_stat = model.Ph(theta)
    else:
        m_stat = -9999.0

    return m_stat

