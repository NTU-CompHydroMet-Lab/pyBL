import pytest
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Tuple
import os
from pybl.timeseries import IndexedShapshot


@pytest.fixture(scope="session")
def rain_timeseries() -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    data_path = os.path.join(os.path.dirname(__file__), "data", "elmdon.csv")
    data = pd.read_csv(data_path, parse_dates=["datatime"])
    data["datatime"] = data["datatime"].astype("int64") // 10**9
    time = data["datatime"].to_numpy()
    intensity = data["Elmdon"].to_numpy()
    return time, intensity
