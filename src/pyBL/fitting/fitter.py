from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import numba as nb
from scipy.optimize import dual_annealing

from pyBL.models import Stat_Props, BLRPRx, BLRPRx_params
from pyBL.models.blrprx import _blrprx_covariance, _blrprx_mean, _blrprx_variance, _blrprx_moment_3rd, _blrprx_kernel
from pyBL.raincell import IConstantRCI, ExponentialRCIModel

class BLRPRxFitter:
    __slots__ = (
        "_model",
        "_props",
        "_props_size",
        "_enable_props_size",
        "_timescales",
        "_mask",
    )

    def __init__(
        self,
        props: Optional[List[Stat_Props]] = None,
        timescales: Optional[npt.NDArray] = None,
        mask: Optional[npt.NDArray] = None,
    ):
        self._props_size = len(Stat_Props)
        self._props = {member: False for member in Stat_Props}

        if props is not None and not all(
            isinstance(prop, Stat_Props) for prop in props
        ):
            raise ValueError("All elements in props must be of type Stat_Props")
        elif props is not None and len(props) == 0:
            raise ValueError("props must not be empty")
        elif props is not None:
            for prop in props:
                self._props[prop] = True
        else:
            for prop in [
                Stat_Props.MEAN,
                Stat_Props.CVAR,
                Stat_Props.AR1,
                Stat_Props.SKEWNESS,
                Stat_Props.pDRY,
            ]:
                self._props[prop] = True

        self._enable_props_size = sum(self._props.values())

        if timescales is not None and not all(
            isinstance(ts, (float, int)) for ts in timescales
        ):
            raise ValueError("All elements in timescale must be of type float")
        elif timescales is not None and len(timescales) == 0:
            raise ValueError("timescales must not be empty")
        elif timescales is not None:
            self._timescales = np.array(timescales, dtype=float)
        else:
            self._timescales = np.array(
                [1, 3, 6, 24], dtype=float
            )  # 1hr, 3hr, 6hr, 24hr

        # Creating mask for togglable properties combination
        if props is None and timescales is None and mask is None:
            self._mask = np.zeros((len(self._timescales), len(self._props)), dtype=bool)
            self._mask[0, 0] = True
            self._mask[:, [1, 2, 3]] = True
        elif mask is None:
            print(
                "Props or Timescales have been set. But mask is not set. All toggled props will be True for all timescales."
            )
            self._mask = np.zeros((len(self._timescales), len(self._props)), dtype=bool)
            # Toggle props that are set to True
            for i, prop in enumerate(self._props):
                if self._props[prop]:
                    self._mask[:, i] = True
        else:
            if mask.shape != (len(self._timescales), len(self._props)):
                raise ValueError("Mask must be of shape (len(props), len(timescales))")
            self._mask = mask

    def template(self):
        pass

    @property
    def props(self):
        return self._props

    # This method decide which properties in Stat_Props to use for fitting
    def set_props(
        self,
        props: Union[List[Stat_Props], Stat_Props],
        mask: Optional[npt.ArrayLike] = None,
    ):
        try:
            mask = np.array(mask, dtype=bool)
        except:
            raise ValueError("Mask must can be converted to numpy array of dtype=bool")

        if isinstance(props, Stat_Props):
            props = [props]

        if not all(isinstance(prop, Stat_Props) for prop in props):
            raise ValueError("All elements in props must be of type Stat_Props")
        if mask is None:
            for prop in props:
                self._props[prop] = True
                self._mask[:, prop.value] = True
        elif mask is not None:
            flag = True
            if mask.ndim == 0:
                if mask is False:
                    flag = False
                mask = np.stack([mask] * len(self._timescales), axis=0)
            if mask.ndim == 1:
                mask = np.stack([mask] * len(props), axis=0)
            if mask.shape != (len(props), len(self._timescales)):
                raise ValueError(
                    "Mask must be of shape (len(input_props), len(timescales))"
                )
            for i, prop in enumerate(props):
                self._props[prop] = flag
                self._mask[:, prop.value] = mask[i, :]

        self._enable_props_size = sum(self._props.values())

    @property
    def timescales(self):
        return self._timescales

    def set_timescale(
        self, timescales: npt.ArrayLike, mask: Optional[npt.ArrayLike] = None
    ):
        if mask is not None:
            try:
                mask = np.array(mask, dtype=bool)
            except:
                raise ValueError(
                    "Mask must can be converted to numpy array of dtype=bool"
                )

        try:
            timescales = np.array(timescales, ndmin=1, dtype=float)
        except:
            raise ValueError("Timescales must be of type float or list of type floats")

        if mask is None:
            # Find if _timescales exist in self._timescales
            for ts in timescales:
                if ts not in self._timescales:
                    self._timescales = np.append(self._timescales, ts)
                    self._mask = np.append(
                        self._mask, np.zeros((1, len(self._props)), dtype=bool), axis=0
                    )

            # Get all timescales index that need to be update
            ts_idx = []
            for ts in timescales:
                ts_idx.append(np.where(self._timescales == ts)[0][0])

            # Toggle all props that is true in self._props
            for prop in self._props:
                if self._props[prop]:
                    self._mask[ts_idx, prop.value] = True
        else:
            # TODO: Implement if mask is not None
            pass

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask: npt.NDArray):
        """
        Mask is a 2D array of shape (len(props), len(timescales)).
        It is used to toggle which properties to use for fitting score calculation.
        """
        if mask.shape != (len(self._props), len(self._timescales)):
            raise ValueError("Mask must be of shape (len(props), len(timescales))")
        self._mask = np.array(mask, dtype=bool)

        # Compress mask along axis=0 with logical or
        for i, prop in enumerate(self._props):
            self._props[prop] = np.any(self._mask[:, i])

    def fit(
        self,
        target: npt.ArrayLike,
        weight: npt.ArrayLike,
        model: Optional[BLRPRx] = None,
    ):
        target_np = np.array(target, dtype=float)
        weight_np = np.array(weight, dtype=float)

        if target_np.shape != weight_np.shape:
            raise ValueError("Target and weight must have the same shape")

        # Check if target and weight are 2D array that have the same shape as mask
        if target_np.ndim != 2:
            raise ValueError("Target must be a 2D array.")

        if target_np.shape != (len(self._timescales), self._enable_props_size):
            raise ValueError(
                "Target must be of shape (len(timescales), len(enabled_props))"
            )

        # Extract column of enabled props in mask.
        enabled_mask = self._mask[
            :, [prop.value for prop in self._props if self._props[prop]]
        ]
        # Apply mask on weight
        weight_np = weight_np * enabled_mask

        if model is None:
            model = BLRPRx()
        else:
            model = model.copy()

        x0 = model.get_params()

        result = dual_annealing(
            self._evaluate,
            bounds=[(0.001, 20)] * x0.size(),
            x0=x0.unpack(),
            maxiter=1000,
            args=(target_np, weight_np, model),
            seed=0,
        )
        BLRPRx_params(*result.x)
        return result, BLRPRx(BLRPRx_params(*result.x), rci_model=model.rci_model)

    def evaluate(
        self,
        x: BLRPRx_params,
        target: npt.ArrayLike,
        weight: npt.ArrayLike,
        model: Optional[BLRPRx] = None,
    ):
        if model is None:
            model = BLRPRx()
        else:
            model = model.copy()

        x_ = np.array(x.unpack(), dtype=float)
        weight_np = np.array(weight, dtype=float)

        # Extract column of enabled props in mask.
        enabled_mask = self._mask[
            :, [prop.value for prop in self._props if self._props[prop]]
        ]
        # Apply mask on weight
        weight_np = weight_np * enabled_mask

        return self._evaluate(x_, target, weight_np, model)

    def _evaluate(
        self,
        x: npt.NDArray[np.float64],
        target: npt.ArrayLike,
        weight: npt.ArrayLike,
        model: BLRPRx,
    ):
        result = np.zeros((len(self._timescales), self._enable_props_size), dtype=float)
        model.update_params(BLRPRx_params(*x))
        enabled_props = [prop for prop in self._props if self._props[prop]]
        for i, ts in enumerate(self._timescales):
            for prop in enabled_props:
                if self._mask[i, prop.value]:
                    result[i, prop.value] = model.get_prop(prop, ts)
        diff = np.sum(np.power(result - target, 2) * weight)
        return diff

class BLRPRxConfig:
    __slots__ = (
        "_mask",
        "_weight",
        "_target",
        "_rci_model"
    )
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
            raise ValueError("target, weight and mask must have the same set of columns")

        # Check if they have the same set of index regardless of order
        if not (set(target.index) == set(weight.index) == set(mask.index)):
            raise ValueError("target, weight and mask must have the same set of index")
        
        # Make sure there isn't duplicate columns
        if len(target.columns) != len(set(target.columns)) or len(weight.columns) != len(set(weight.columns)) or len(mask.columns) != len(set(mask.columns)):
            raise ValueError("target, weight and mask must not have duplicate columns")

        # Make sure there isn't duplicate index
        if len(target.index) != len(set(target.index)) or len(weight.index) != len(set(weight.index)) or len(mask.index) != len(set(mask.index)):
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

    def get_evaluation_func(self) -> Callable[[npt.NDArray], np.float64]:
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
            lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = x
            f1 = f1_func(sigmax_mux)
            f2 = f2_func(sigmax_mux)
            for t, scale in enumerate(scales):
                mean = _blrprx_mean(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota)
                variance = _blrprx_variance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1)
                moment_3rd = _blrprx_moment_3rd(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, f2)
                #if not mask_np[t, stat]:
                    #continue
                if mask_np[t, 0]:     # Stat_Props.MEAN.value
                    predict[t, 0] = mean
                if mask_np[t,1]:   # Stat_Props.CVAR.value
                    predict[t, 1] = np.sqrt(variance) / mean 
                if mask_np[t,2]:   # Stat_Props.SKEWNESS.value
                    predict[t, 2] = moment_3rd / np.power(variance, 1.5)
                if mask_np[t,3]:   # Stat_Props.AR1.value
                    predict[t, 3] = _blrprx_covariance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, 1.0) / variance
                if mask_np[t,4]:   # Stat_Props.AR2.value
                    predict[t, 4] = _blrprx_covariance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, 2.0) / variance
                if mask_np[t,5]:   # Stat_Props.AR3.value
                    predict[t, 5] = _blrprx_covariance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, 3.0) / variance
                if mask_np[t,6]:   # Stat_Props.pDRY.value
                    predict[t, 6] = 0
                if mask_np[t,7]:   # Stat_Props.MSIT.value
                    predict[t, 7] = 0
                if mask_np[t,8]:   # Stat_Props.MSD.value
                    predict[t, 8] = 0
                if mask_np[t,9]:   # Stat_Props.MCIT.value
                    predict[t, 9] = 0
                if mask_np[t,10]:   # Stat_Props.MCD.value
                    predict[t, 10] = 0
                if mask_np[t,11]:   # Stat_Props.MCS.value
                    predict[t, 11] = 0
                if mask_np[t,12]:   # Stat_Props.MPC.value
                    predict[t, 12] = 0
                if mask_np[t,13]:   # Stat_Props.VAR.value
                    predict[t, 13] = variance
                if mask_np[t,14]:   # Stat_Props.AC1.value
                    predict[t, 14] = _blrprx_covariance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, 1.0)
                if mask_np[t,15]:   # Stat_Props.AC2.value
                    predict[t, 15] = _blrprx_covariance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, 2.0)
                if mask_np[t,16]:   # Stat_Props.AC3.value
                    predict[t, 16] = _blrprx_covariance(scale, lambda_, phi, kappa, alpha, nu, sigmax_mux, iota, f1, 3.0)
            return np.nansum(np.power(predict - target_np, 2) * weight_np * mask_np)
        return evaluation_func

    @staticmethod
    def default_target(scale_list: List[float] = [1]):
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        target_df = pd.DataFrame(columns=[prop for prop in Stat_Props], index = scale_list, dtype=np.float64)
        target_df.index.name = "timescales"
        return target_df

    @staticmethod
    def default_weight(scale_list: List[float] = [1]):
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        weight_df = pd.DataFrame(columns=[prop for prop in Stat_Props], index = scale_list, dtype=np.float64)
        weight_df.index.name = "timescales"
        return weight_df
    
    @staticmethod
    def default_mask(scale_list: List[float] = [1]):
        # Create a numpy array with Stat_Props as columns one row with timescales=1,3,6,24, with default value of False
        mask_np = np.zeros((len(scale_list), len(Stat_Props)), dtype=np.int_)
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24, with default value of False
        mask_df = pd.DataFrame(data=mask_np, columns=[prop for prop in Stat_Props], index = scale_list, dtype=np.int_)
        mask_df.index.name = "timescales"
        return mask_df
    
