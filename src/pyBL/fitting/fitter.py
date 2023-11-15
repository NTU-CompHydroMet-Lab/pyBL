from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import dual_annealing

from pyBL.models import Stat_Props, BLRPRx, BLRPRx_params


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
            self._evauluate,
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

        return self._evauluate(x_, target, weight_np, model)

    def _evauluate(
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

class BLRPRxFitter_pd:
    __slots__ = (
        "_mask",
        "_weight",
        "_target"
    )
    def __init__(
        self,
        mask: pd.DataFrame,
        weight: pd.DataFrame,
        target: pd.DataFrame,
    ):
        # Check the format of mask, weight and target
         
        

    @classmethod
    def template(cls):
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        template = pd.DataFrame(columns=[prop.name for prop in Stat_Props], index=[1,3,6,24], dtype=np.float64)
        return template