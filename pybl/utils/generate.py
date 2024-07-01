from typing import Optional, Union, overload

import numpy as np
import numpy.typing as npt
import pandas as pd


@overload
def generate(intensity: pd.Series[float], sample_size: int) -> pd.Series[float]: ...


@overload
def generate(
    intensity: npt.NDArray[np.float64],
    sample_size: int,
    epoch_time: Optional[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]: ...


def generate(
    intensity: Union[pd.Series[float], npt.NDArray[np.float64]],
    sample_size: int,
    *args,
    **kwargs
) -> Union[pd.Series[float], npt.NDArray[np.float64]]:
    return np.empty(0)
