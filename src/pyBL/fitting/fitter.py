from typing import Callable, List, Optional

import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd

from pyBL.models import Stat_Props
from pyBL.models.blrprx import (
    _blrprx_covariance,
    _blrprx_mean,
    _blrprx_moment_3rd,
    _blrprx_variance,
)
from pyBL.raincell import ExponentialRCIModel, IConstantRCI


class BLRPRxConfig:
    __slots__ = ("_mask", "_weight", "_target", "_rci_model")

    _mask: pd.DataFrame
    _weight: pd.DataFrame
    _target: pd.DataFrame
    _rci_model: IConstantRCI

    def __init__(
        self,
        target: pd.DataFrame,
        weight: pd.DataFrame,
        mask: pd.DataFrame,
        rci_model: Optional[IConstantRCI] = None,
    ):
        if not (target.shape == weight.shape == mask.shape):
            raise ValueError("target, weight and mask must have the same shape")

        # Check if they have the same set of column regardless of order
        if not (set(target.columns) == set(weight.columns) == set(mask.columns)):
            raise ValueError(
                "target, weight and mask must have the same set of columns"
            )

        # Check if they have the same set of index regardless of order
        if not (set(target.index) == set(weight.index) == set(mask.index)):
            raise ValueError("target, weight and mask must have the same set of index")

        # Make sure there isn't duplicate columns
        if (
            len(target.columns) != len(set(target.columns))
            or len(weight.columns) != len(set(weight.columns))
            or len(mask.columns) != len(set(mask.columns))
        ):
            raise ValueError("target, weight and mask must not have duplicate columns")

        # Make sure there isn't duplicate index
        if (
            len(target.index) != len(set(target.index))
            or len(weight.index) != len(set(weight.index))
            or len(mask.index) != len(set(mask.index))
        ):
            raise ValueError("target, weight and mask must not have duplicate index")

        # Check if rci_model is None or IConstantRCI
        if rci_model is None:
            self._rci_model = ExponentialRCIModel()
        elif isinstance(rci_model, IConstantRCI):
            self._rci_model = rci_model
        else:
            raise ValueError("rci_model must follow IConstantRCI protocol")

        self._target = target.copy(deep=True)
        self._weight = weight.copy(deep=True)
        self._mask = mask.copy(deep=True)

    def get_evaluation_func(self) -> Callable[[npt.NDArray[np.float64]], np.float64]:
        '''
        Return a objective function based on your configuration of target, weight and mask.
        It is a function that takes in a numpy array of 7 parameters and return a score.

        Note that the first time this function is called, it will take a while to compile.

        Returns:
        --------
        evaluation_func: Callable[[npt.NDArray[np.float64]], np.float64]
            A function that takes in a numpy array of 7 parameters and return a error score.
        '''
        scales = self._target.index.to_numpy()
        stats_len = len(Stat_Props)

        target_np = self._target.to_numpy()
        weight_np = self._weight.to_numpy()
        mask_np = self._mask.to_numpy().astype(np.bool_)

        f1_func = self._rci_model.get_f1
        f2_func = self._rci_model.get_f2

        @nb.njit
        def evaluation_func(x: npt.NDArray[np.float64]) -> np.float64:
            predict = np.zeros((len(scales), stats_len), dtype=np.float64)
            # fmt: off
            (l ,p ,k ,a ,n ,s ,i) = x  # noqa: E741.   lambda, phi, kappa, alpha, nu, sigma, iota
            # fmt: on
            f1 = f1_func(s)
            f2 = f2_func(s)
            score = np.float64(0)
            for t, scale in enumerate(scales):
                mean = _blrprx_mean(scale, l, p, k, a, n, s, i)
                variance = _blrprx_variance(scale, f1, l, p, k, a, n, s, i)
                moment_3rd = _blrprx_moment_3rd(scale, f1, f2, l, p, k, a, n, s, i)
                # if not mask_np[t, stat]:
                # continue
                if mask_np[t, 0]:  # Stat_Props.MEAN.value
                    predict[t, 0] = mean
                    score += (
                        np.power(predict[t, 0] - target_np[t, 0], 2) * weight_np[t, 0]
                    )
                if mask_np[t, 1]:  # Stat_Props.CVAR.value
                    predict[t, 1] = np.sqrt(variance) / mean
                    score += (
                        np.power(predict[t, 1] - target_np[t, 1], 2) * weight_np[t, 1]
                    )
                if mask_np[t, 2]:  # Stat_Props.SKEWNESS.value
                    predict[t, 2] = moment_3rd / np.power(variance, 1.5)
                    score += (
                        np.power(predict[t, 2] - target_np[t, 2], 2) * weight_np[t, 2]
                    )
                if mask_np[t, 3]:  # Stat_Props.AR1.value
                    predict[t, 3] = (
                        _blrprx_covariance(scale, f1, 1.0, l, p, k, a, n, s, i)
                        / variance
                    )
                    score += (
                        np.power(predict[t, 3] - target_np[t, 3], 2) * weight_np[t, 3]
                    )
                if mask_np[t, 4]:  # Stat_Props.AR2.value
                    predict[t, 4] = (
                        _blrprx_covariance(scale, f1, 2.0, l, p, k, a, n, s, i)
                        / variance
                    )
                    score += (
                        np.power(predict[t, 4] - target_np[t, 4], 2) * weight_np[t, 4]
                    )
                if mask_np[t, 5]:  # Stat_Props.AR3.value
                    predict[t, 5] = (
                        _blrprx_covariance(scale, f1, 3.0, l, p, k, a, n, s, i)
                        / variance
                    )
                    score += (
                        np.power(predict[t, 5] - target_np[t, 5], 2) * weight_np[t, 5]
                    )
                if mask_np[t, 6]:  # Stat_Props.pDRY.value
                    predict[t, 6] = 0
                    score += (
                        np.power(predict[t, 6] - target_np[t, 6], 2) * weight_np[t, 6]
                    )
                if mask_np[t, 7]:  # Stat_Props.MSIT.value
                    predict[t, 7] = 0
                    score += (
                        np.power(predict[t, 7] - target_np[t, 7], 2) * weight_np[t, 7]
                    )
                if mask_np[t, 8]:  # Stat_Props.MSD.value
                    predict[t, 8] = 0
                    score += (
                        np.power(predict[t, 8] - target_np[t, 8], 2) * weight_np[t, 8]
                    )
                if mask_np[t, 9]:  # Stat_Props.MCIT.value
                    predict[t, 9] = 0
                    score += (
                        np.power(predict[t, 9] - target_np[t, 9], 2) * weight_np[t, 9]
                    )
                if mask_np[t, 10]:  # Stat_Props.MCD.value
                    predict[t, 10] = 0
                    score += (
                        np.power(predict[t, 10] - target_np[t, 10], 2)
                        * weight_np[t, 10]
                    )
                if mask_np[t, 11]:  # Stat_Props.MCS.value
                    predict[t, 11] = 0
                    score += (
                        np.power(predict[t, 11] - target_np[t, 11], 2)
                        * weight_np[t, 11]
                    )
                if mask_np[t, 12]:  # Stat_Props.MPC.value
                    predict[t, 12] = 0
                    score += (
                        np.power(predict[t, 12] - target_np[t, 12], 2)
                        * weight_np[t, 12]
                    )
                if mask_np[t, 13]:  # Stat_Props.VAR.value
                    predict[t, 13] = variance
                    score += (
                        np.power(predict[t, 13] - target_np[t, 13], 2)
                        * weight_np[t, 13]
                    )
                if mask_np[t, 14]:  # Stat_Props.AC1.value
                    predict[t, 14] = _blrprx_covariance(
                        scale, f1, 1.0, l, p, k, a, n, s, i
                    )
                    score += (
                        np.power(predict[t, 14] - target_np[t, 14], 2)
                        * weight_np[t, 14]
                    )
                if mask_np[t, 15]:  # Stat_Props.AC2.value
                    predict[t, 15] = _blrprx_covariance(
                        scale, f1, 2.0, l, p, k, a, n, s, i
                    )
                    score += (
                        np.power(predict[t, 15] - target_np[t, 15], 2)
                        * weight_np[t, 15]
                    )
                if mask_np[t, 16]:  # Stat_Props.AC3.value
                    predict[t, 16] = _blrprx_covariance(
                        scale, f1, 3.0, l, p, k, a, n, s, i
                    )
                    score += (
                        np.power(predict[t, 16] - target_np[t, 16], 2)
                        * weight_np[t, 16]
                    )
            # return np.nansum(np.power(predict - target_np, 2) * weight_np * mask_np)
            return score

        return evaluation_func

    @staticmethod
    def default_target(timescale: List[float] = [1, 3, 6, 24]) -> pd.DataFrame:
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        target_df = pd.DataFrame(
            columns=[prop for prop in Stat_Props], index=timescale, dtype=np.float64
        )
        target_df.index.name = "timescales"

        return target_df

    @staticmethod
    def default_weight(timescale: List[float] = [1, 3, 6, 24]) -> pd.DataFrame:
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        weight_df = pd.DataFrame(
            columns=[prop for prop in Stat_Props], index=timescale, dtype=np.float64
        )
        weight_df.index.name = "timescales"
        return weight_df

    @staticmethod
    def default_mask(timescale: List[float] = [1, 3, 6, 24]) -> pd.DataFrame:
        # Create a numpy array with Stat_Props as columns one row with timescales=1,3,6,24, with default value of False
        mask_np = np.zeros((len(timescale), len(Stat_Props)), dtype=np.int_)
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24, with default value of False
        mask_df = pd.DataFrame(
            data=mask_np,
            columns=[prop for prop in Stat_Props],
            index=timescale,
            dtype=np.int_,
        )
        mask_df.index.name = "timescales"
        return mask_df
