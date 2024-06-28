import numpy as np
import numpy.typing as npt
from typing import Union, overload, Optional
import pandas as pd

@overload
def generate(intensity: pd.Series, sample_size: int) -> pd.Series: ...

@overload
def generate(intensity: npt.NDArray, sample_size: int, epoch_time: Optional[npt.NDArray]) -> npt.NDArray: ...

def generate(intensity: Union[pd.Series, npt.NDArray], sample_size: int, *args, **kwargs) -> Union[pd.Series, npt.NDArray]:
    ...