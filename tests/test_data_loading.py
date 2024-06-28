import numpy.typing as npt
import numpy as np
import pandas as pd
from typing import Tuple


def test_show_data(
    elmdon_data_np: Tuple[npt.NDArray, npt.NDArray],
    elmdon_data_pd: pd.Series,
    bochum_data_np: Tuple[npt.NDArray, npt.NDArray],
    bochum_data_pd: pd.Series,
) -> None:
    elm_time, elm_intensity = elmdon_data_np
    boc_time, boc_intensity = bochum_data_np

    assert elm_intensity.mean() == elmdon_data_pd.mean()
    assert np.nanmean(boc_intensity) == bochum_data_pd.mean()
    assert elm_intensity.sum() == elmdon_data_pd.sum()
    assert np.nansum(boc_intensity) == bochum_data_pd.sum()
    assert np.isclose(
        np.cov(elm_intensity), elmdon_data_pd.cov(elmdon_data_pd), atol=1e-20
    )
    assert np.isclose(
        np.nanvar(boc_intensity), bochum_data_pd.cov(bochum_data_pd), atol=1e-20
    )
