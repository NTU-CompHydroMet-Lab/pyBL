import pytest
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple
import os
from pybl.utils.timeseries import preprocess_classic


@pytest.fixture(scope="session")
def elmdon_data_np() -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Elmdon data doesn't contain any missing values.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "elmdon.csv")
    data = pd.read_csv(data_path, parse_dates=["datatime"])
    data["datatime"] = data["datatime"].astype("int64") // 10**9
    time = data["datatime"].to_numpy()
    intensity = data["Elmdon"].to_numpy()
    return time, intensity


@pytest.fixture(scope="session")
def elmdon_data_pd() -> pd.Series:
    data_path = os.path.join(os.path.dirname(__file__), "data", "elmdon.csv")
    data = pd.read_csv(data_path, parse_dates=["datatime"], index_col="datatime")
    # Replace -1 with np.nan
    data["Elmdon"] = data["Elmdon"].replace(-1, np.nan)
    return data["Elmdon"]


@pytest.fixture(scope="session")
def elmdon_stats_weight(
    bochum_data_np,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Statistical Properties and Weights for Elmdon data.
    """
    stats = np.load(os.path.join(os.path.dirname(__file__), "data", "elmdon_stats.npy"))
    weight = np.load(
        os.path.join(os.path.dirname(__file__), "data", "elmdon_weight.npy")
    )
    return stats, weight


@pytest.fixture(scope="session")
def bochum_data_np() -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Bochum data contains missing values (-1). All missing values are replaced with np.nan.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "bochum_5min.csv")
    data = pd.read_csv(data_path, parse_dates=["datatime"])
    data["datatime"] = data["datatime"].astype("int64") // 10**9
    time = data["datatime"].to_numpy()
    intensity = data[list(data.keys())[-1]].to_numpy()
    # Replace -1 with np.nan
    intensity = np.where(intensity == -1, np.nan, intensity)
    return time, intensity


@pytest.fixture(scope="session")
def bochum_data_pd() -> pd.Series:
    """
    Bochum data contains missing values (-1). All missing values are replaced with np.nan.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "bochum_5min.csv")
    data = pd.read_csv(data_path, parse_dates=["datatime"], index_col="datatime")
    # Replace -1 with np.nan
    data["Bochum"] = data["Bochum"].replace(-1, np.nan)
    return data["Bochum"]


@pytest.fixture(scope="session")
def bochum_stats_weight(
    bochum_data_np,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Statistical Properties and Weights for Bochum data.
    """
    stats = np.load(os.path.join(os.path.dirname(__file__), "data", "bochum_stats.npy"))
    weight = np.load(os.path.join(os.path.dirname(__file__), "data", "bochum_weight.npy"))
    return stats, weight

@pytest.fixture(scope="session")
def fuzzy_short_data() -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Generate a fuzzy time series data.

    return:
    --------------
        data: npt.NDArray[np.float64]
            A fuzzy time series data with the shape of (N, 2, T).
    """
    rng = np.random.RandomState(0)
    N = 10000
    data = []
    for i in range(N):
        length = rng.randint(2, 10)

        time_interval = rng.randint(1, 100, size=1)[0]
        time_start = rng.randint(-length, length, size=1)[0]
        time = np.arange(time_start, time_start+length)*time_interval

        intensity = rng.rand(length)
        nan_ratio = rng.rand()
        nan_index = rng.choice(np.arange(length), size=int(length*nan_ratio), replace=False)
        intensity[nan_index] = np.nan

        data.append((time, intensity))
    
    return data

@pytest.fixture(scope="session")
def fuzzy_long_data() -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Generate a fuzzy time series data.

    return:
    --------------
        data: npt.NDArray[np.float64]
            A fuzzy time series data with the shape of (N, 2, T).
    """
    rng = np.random.RandomState(0)
    N = 100
    data = []
    for i in range(N):
        length = rng.randint(1000, 10000)

        time_interval = rng.randint(1, 100, size=1)

        intensity = rng.rand(length)
        nan_ratio = rng.rand()
        nan_index = rng.choice(np.arange(length), size=int(length*nan_ratio), replace=False)
        intensity[nan_index] = np.nan

        data.append((np.arange(length)*time_interval, intensity))
    
    return data
