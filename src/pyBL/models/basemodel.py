from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Type, Union

import numpy as np
import scipy as sp  # type: ignore
import math

from pyBL.raincell import ExponentialRCIModel, IConstantRCI

BaseBLRP_RCIModel = Optional[Union[IConstantRCI, Type[IConstantRCI]]]


class Stat_Props(Enum):
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


class BaseBLRP(Protocol):
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
        elif isinstance(rci_model(), IConstantRCI):
            self.rci_model = rci_model()
        else:
            raise TypeError("rci_model must be a implementation of IConstantRCI")


@dataclass
class BLRPR_params:
    # Rodriguez-Iturbe et al., 1988; Onof and Wheater, 1993
    lambda_: float  # Storm's arrival rate (per hour) parameter of Poisson distribution
    phi: float  # Storm's ratio of expected duration
    kappa: float  # Cell's ratio of arrival rate (per storm) parameter of Poisson distribution
    alpha: float  # Cell's expected duration's shape parameter of Gamma distribution
    nu: float  # Cell's expected duration's scale parameter of Gamma distribution
    mu_x: float  # Cell's expected depth parameter of Gamma TODO:(or exponential) distribution


class BLRPR(BaseBLRP):
    __slots__ = (
        "rci_model",
        "rng",
        "params",
        "lambda_",
        "phi",
        "kappa",
        "alpha",
        "nu",
        "mu_x",
    )

    def __init__(
        self,
        params: Optional[BLRPR_params] = None,
        rng: Optional[np.random.Generator] = None,
        rci_model: BaseBLRP_RCIModel = None,
    ) -> None:
        super().__init__(rng, rci_model)

        self.lambda_: float = params.lambda_ if params is not None else 0.1
        self.phi: float = params.phi if params is not None else 0.1
        self.kappa: float = params.kappa if params is not None else 0.1
        self.alpha: float = params.alpha if params is not None else 3
        self.nu: float = params.nu if params is not None else 0.1
        self.mu_x: float = params.mu_x if params is not None else 0.1

    def _kernel(self, k: float, x: float, nu: float, alpha: float) -> float:
        if alpha <= 4.0 and np.modf(alpha)[0] == 0.0:
            alpha += 1.0e-8

        if alpha - k >= 171.0 or alpha >= 171.0:
            return np.inf

        return (
            np.power(nu / (nu + x), alpha)
            * np.power(nu + x, k)
            * math.gamma(alpha - k)
            / math.gamma(alpha)
        )

    def mean(self, timescale: float) -> float:
        # TODO: Check the formula. Why do we have a kernel here?
        return (
            timescale
            * self.lambda_
            * self.mu_x
            * (1.0 + self.kappa / self.phi)
            * self._kernel(1.0, 0, self.nu, self.alpha)
        )
