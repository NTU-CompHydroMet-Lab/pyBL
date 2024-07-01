from __future__ import annotations

from enum import Enum
from typing import Optional, Type, Union

import numpy as np

from pybl.raincell import ExponentialRCIModel, IConstantRCI

BaseBLRP_RCIModel = Optional[Union[IConstantRCI, Type[IConstantRCI]]]


class StatMetrics(Enum):
    MEAN = 0  # Mean
    CVAR = 1  # Coefficient of variation
    SKEWNESS = 2  # Skewness
    AR1 = 3  # Autocorrelation coefficient lag 1
    AR2 = 4  # Autocorrelation coefficient lag 2
    AR3 = 5  # Autocorrelation coefficient lag 3
    pDRY = 6  # Probability of dry
    MSIT = 7
    MSD = 8
    MCIT = 9
    MCD = 10
    MCS = 11
    MPC = 12
    VAR = 13
    AC1 = 14  # Autocorrelation lag 1 (Just covariance with different lags)
    AC2 = 15  # Autocorrelation lag 2 (Just covariance with different lags)
    AC3 = 16  # Autocorrelation lag 3 (Just covariance with different lags)


class BaseBLRP():
    rci_model: IConstantRCI
    rng: np.random.Generator

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        rci_model: BaseBLRP_RCIModel = None,
    ) -> None:
        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)  # type: ignore

        if rci_model is None:
            self.rci_model = ExponentialRCIModel()
        elif isinstance(rci_model, IConstantRCI):
            self.rci_model = rci_model
        else:
            raise TypeError("rci_model must be a implementation of IConstantRCI")
