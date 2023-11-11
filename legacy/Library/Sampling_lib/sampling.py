from datetime import datetime as dt
from datetime import timedelta as td

import numpy as np
import pandas as pd
import yaml
from dotmap import DotMap
from Library.BLRPRmodel.BLRPRx import *
from Library.Cal_stats_lib.utils.stats_calculation import *
from Library.Cal_stats_lib.utils.utils import *
from Library.Sampling_lib.mergeCells import *
from Library.Sampling_lib.sampling import *
from scipy.stats import poisson

from .storm import *

gamma = np.random.gamma

args = yaml.load(open("./config/default.yaml"), Loader=yaml.FullLoader)
args = DotMap(args)
start_time = args.sampling.start_time
end_time = args.sampling.end_time


def SampleIntensity(para_ins, rng_seed):  # cell intensity (mm/h)
    if len(para_ins) < 2:
        return -9999.0
    # TODO :check sigmax_mux
    # sigmax_mux = 1.0??????
    mux, sigmax_mux = para_ins[0], para_ins[1]

    I_shape = 1.0 / sigmax_mux / sigmax_mux
    I_scale = sigmax_mux * sigmax_mux * mux
    return rng_seed.gamma(I_shape, scale=I_scale, size=1)[0]


def SampleStorm(thetas, sDT_sim_hr, duration_sim_hr):
    # all operations are working with unit = hour
    ld = thetas[0]  # storm arrival rate
    iota = thetas[1]  # ratio of mean cell intensity to eta
    alpha = thetas[2]  # shape parameter for gamma distribution of eta
    nu = alpha / thetas[3]  # scale parameter for gamma distribution
    kappa = thetas[4]  # cell arrival rate
    phi = thetas[5]  # storm termination rate

    # the number of storms within the 'simulation' period - a Poisson distributed random number
    rng_seed = np.random.RandomState(100000)
    # n_storms = rng_seed.poisson(ld * duration_sim_hr, size=1)[0]
    n_storms = poisson.rvs(ld * duration_sim_hr)
    print("%d samples of storms are generated. \n" % n_storms)

    storms = list()

    # single storm simulation
    for s in range(n_storms):
        # cell duration parameter
        # eta = etas[s]
        eta = gamma(alpha, 1.0 / nu)
        # storm termination rate
        gama = phi * eta
        # cell arrival rate
        beta = kappa * eta
        # mean cell intensity
        mux = iota * eta

        # Sample storm duration - an exponential random number
        duration_storm_hr = rng_seed.exponential(scale=1.0 / gama, size=1)[0]

        # number of cells within a storm - a Poisson distributed random number
        n_cells = 1 + rng_seed.poisson(beta * duration_storm_hr, size=1)[0]
        storm = Storm(n_cells)

        # starting time of a storm - a uniformly random number
        sDT_storm_hr = sDT_sim_hr + td(
            hours=duration_sim_hr * rng_seed.uniform(size=1)[0]
        )

        start_time = args.sampling.start_time

        ### =============================================================== duration strom hour ======================================
        try:
            eDT_storm_hr = sDT_storm_hr + td(hours=duration_storm_hr)
        except:
            eDT_storm_hr = dt.strptime(end_time, "%Y-%m-%d") + td(days=90)
            duration_storm_hr = (eDT_storm_hr - sDT_storm_hr).total_seconds() / 3600
        ### =============================================================== duration strom hour ======================================

        storm.sDT, storm.eDT = sDT_storm_hr, eDT_storm_hr

        # Sample for the first cell
        storm.cells[0] = Cell()
        # duration - an exponential random number
        duration_cell_hr = rng_seed.exponential(scale=1.0 / eta, size=1)[0]
        # assign start time and end time of cells
        # TODO : hr to second
        storm.cells[0].sDT = sDT_storm_hr

        ### ==================================================== cell end time =================================================
        try:
            storm.cells[0].eDT = storm.cells[0].sDT + td(hours=duration_cell_hr)
        except:
            storm.cells[0].eDT = dt.strptime(end_time, "%Y-%m-%d") + td(days=90)
        ### ==================================================== cell end time =================================================

        # cell intensity - a gamma (or exponential) distributed random number
        sigmax_mux = 1.0
        para_xi = [mux, sigmax_mux]
        storm.cells[0].Depth = SampleIntensity(para_xi, rng_seed)  # unit = mm/h

        # Assign date time and intensity to the rest of cells
        # Note: the starting time of the first cell shall be
        # the same as the starting time of the storm
        for c in range(1, n_cells):
            storm.cells[c] = Cell()
            # TODO: check the unit of time
            storm.cells[c].sDT = sDT_storm_hr + td(
                hours=duration_storm_hr * rng_seed.uniform(size=1)[0]
            )

            # Sample cell duration
            duration_cell_hr = rng_seed.exponential(scale=1.0 / eta, size=1)[0]

            ### ==================================================== duration cell hour =================================================
            if duration_cell_hr > 20000000:
                duration_cell_hr = (
                    dt.strptime(end_time, "%Y-%m-%d")
                    - dt.strptime(start_time, "%Y-%m-%d")
                ).total_seconds() / 3600

            if storm.cells[c].sDT > dt.strptime(end_time, "%Y-%m-%d"):
                storm.cells[c].sDT = dt.strptime(end_time, "%Y-%m-%d")
                storm.cells[c].eDT = dt.strptime(end_time, "%Y-%m-%d") + td(days=90)

            storm.cells[c].eDT = storm.cells[c].sDT + td(hours=duration_cell_hr)
            ### ==================================================== duration cell hour =================================================

            storm.cells[c].Depth = SampleIntensity(para_xi, rng_seed)  # unit = mm/h

        storms.append(storm)

    return alpha, nu, phi, storms


def sample_Sta_prop(sub_data, propFile_path, pDryThreshold):
    data = {}
    prop = pd.read_csv(propFile_path, index_col=0, header=None)
    propertyList = prop.loc["prop"].dropna().to_numpy()
    timeScaleList = prop.loc["time"].dropna().to_numpy()
    meanTimeScale = prop.loc["timeForMean"].dropna().to_numpy()
    outputMean = prop.loc["OutputTimeScaleForMean"].dropna().to_numpy()

    for sinmeanTimeScale in meanTimeScale:
        subSeq = change_timescale(sub_data, sinmeanTimeScale)
        col_data = []
        mData = subSeq.mean()
        print(f"scale, mData: {sinmeanTimeScale}, {mData}")
        col_data.append(round(mData, 7))
        if sinmeanTimeScale == outputMean:
            data["{}_{}".format("Mean", str(sinmeanTimeScale))] = col_data

    # step 5 calculate other properties
    for sinTimescale in timeScaleList:
        scaledData = change_timescale(sub_data, sinTimescale)
        for sinStaProp in propertyList[:-1]:
            col_data = []
            input = scaledData.to_numpy().flatten()
            if sinStaProp == "CV":
                ret = cal_CVAR(input)
            elif sinStaProp == "AR-1":
                ret = cal_AR1(input)
            elif sinStaProp == "Skewness":
                ret = cal_skewness(input)
            #             elif sinStaProp == 'pDry':
            #                 ret = cal_pDry(input, pDryThreshold)
            col_data.append(round(ret, 7))
            data["{}_{}".format(str(sinStaProp), str(sinTimescale))] = col_data

    return data
