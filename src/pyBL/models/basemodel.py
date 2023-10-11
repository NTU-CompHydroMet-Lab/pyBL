from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore

from pyBL.raincell import ExponentialRCIModel, IConstantRCI

BaseBLRP_RCIModel = Optional[Union[IConstantRCI, Type[IConstantRCI]]]


class BL_Props(Enum):
    MEAN = 0
    CVAR = 1
    SKEWNESS = 2
    AR1 = 3
    AR2 = 4
    AR3 = 5
    pDRY = 6
    MSIT = 7
    MSD = 8
    MCIT = 9
    MCD = 10
    MCS = 11
    MPC = 12
    VAR = 13
    AC1 = 14
    AC2 = 15
    AC3 = 16


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
            self.rci_model = ExponentialRCIModel(self.rng)
        elif isinstance(rci_model, IConstantRCI):
            self.rci_model = rci_model
        elif isinstance(rci_model(), IConstantRCI):
            self.rci_model = rci_model(self.rng)
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


@dataclass
class BLRPRx_params:
    # Kaczmarska et al. (2014)
    lambda_: float  # Storm's arrival rate (per hour) parameter of Poisson distribution
    phi: float  # ratio of the storm (cell process) termination rate to η (i.e. γ/η)j
    kappa: float  # ratio of the cell arrival rate to η (i.e. β/η)
    alpha: float  # shape parameter for the Gamma distribution of the cell duration parameter, η
    nu: float  # scale parameter for the Gamma distribution of the cell duration parameter, η
    sigmax_mux: float  # ratio of the standard deviation of cell intensity to the mean cell intensity
    iota: float  # ratio of mean cell intensity to η (i.e. µ_x /η)


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
        self.alpha: float = params.alpha if params is not None else 0.1
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
            * sp.special.gamma(alpha - k)
            / sp.special.gamma(alpha)
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


class BLRPRx(BaseBLRP):
    ## This BLRPRx is an implementation of Kaczmarska et al. (2014)
    __slots__ = (
        "rci_model",
        "rng",
        # "params",
        "lambda_",
        "phi",
        "kappa",
        "alpha",
        "nu",
        "sigmax_mux",
        "iota",
    )

    def __init__(
        self,
        params: Optional[BLRPRx_params] = None,
        rng: Optional[np.random.Generator] = None,
        rci_model: BaseBLRP_RCIModel = None,
    ) -> None:
        super().__init__(rng, rci_model)
        # self.params: BLRPRx_params = BLRPRx_params(
        #    lambda_=0.1, phi=0.1, kappa=0.1, alpha=0.1, nu=0.1, sigmax_mux=0.1, iota=0.1,
        # )

        # If user does not provide params, give the default values.
        self.lambda_: float = params.lambda_ if params is not None else 0.1
        self.phi: float = params.phi if params is not None else 0.1
        self.kappa: float = params.kappa if params is not None else 0.1
        self.alpha: float = params.alpha if params is not None else 0.1
        self.nu: float = params.nu if params is not None else 0.1
        self.sigmax_mux: float = params.sigmax_mux if params is not None else 0.1
        self.iota: float = params.iota if params is not None else 0.1

    def unpack_params(
        self, params: Optional[BLRPRx_params]
    ) -> Tuple[float, float, float, float, float, float, float]:
        # If user does not provide params, use the params in the class.
        if params is None:
            return (
                self.lambda_,
                self.phi,
                self.kappa,
                self.alpha,
                self.nu,
                self.sigmax_mux,
                self.iota,
            )
        else:
            return (
                params.lambda_,
                params.phi,
                params.kappa,
                params.alpha,
                params.nu,
                params.sigmax_mux,
                params.iota,
            )

    def _kernel(self, k: float, x: float, nu: float, alpha: float) -> float:
        # TODO: This is just for numerical stability. Check the original paper.
        if alpha <= 4.0 and np.modf(alpha)[0] == 0.0:
            alpha += 1.0e-8

        if alpha - k >= 171.0 or alpha >= 171.0:
            return np.inf

        return (
            np.power(nu / (nu + x), alpha)
            * np.power(nu + x, k)
            * sp.special.gamma(alpha - k)
            / sp.special.gamma(alpha)
        )

    def mean(self, timescale: float, params: Optional[BLRPRx_params]) -> float:
        lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = self.unpack_params(params)
        mu_c = 1.0 + kappa / phi

        # TODO: Check the formula. Why do we have a kernel here?
        return timescale * lambda_ * iota * mu_c * self._kernel(1.0, 0, nu, alpha)

    def variance(self, timescale: float, params: Optional[BLRPRx_params]) -> float:
        lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = self.unpack_params(params)
        mu_c = 1.0 + kappa / phi
        f1 = self.rci_model.get_f1(sigmax_mux=sigmax_mux)

        # TODO: The formula is not the same as C/C++ code. Check this.
        # The formula is the same as Kaczmarska et al. (2014) A.2
        var_part1 = timescale * (f1 + kappa / phi)
        var_part2 = (
            kappa * (1 - phi**3) / (phi**4 - phi**2) - f1
        ) * self._kernel(1.0, 0, nu, alpha)
        var_part3 = (kappa / (phi**4 - phi**2)) * self._kernel(
            1.0, phi * timescale, nu, alpha
        )
        var_part4 = (kappa * phi / (phi**2 - 1)) * self._kernel(
            1.0, timescale, nu, alpha
        )

        return (
            2
            * lambda_
            * mu_c
            * iota**2
            * (var_part1 + var_part2 - var_part3 + var_part4)
        )

    def covariance(
        self, timescale: float, lag: float, params: Optional[BLRPRx_params]
    ) -> float:
        lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = self.unpack_params(params)
        mu_c = 1.0 + kappa / phi
        f1 = self.rci_model.get_f1(sigmax_mux=sigmax_mux)

        # TODO: In the original code there's a variable k = 3.0 - 2.0 * c.
        # Is it the same as # Kaczmarska et al. (2014) A.10?
        # The lag between covariance? Seems it doesn't.
        # Because in Kaczmarska et al. (2014) (1) There is also a k. That is not lag right?

        cov_part1 = (f1 + (kappa * phi) / (phi**2 - 1)) * (
            self._kernel(1.0, (lag - 1.0) * timescale, nu, alpha)
            - 2 * self._kernel(1.0, lag * timescale, nu, alpha)
            + self._kernel(1.0, (lag + 1.0) * timescale, nu, alpha)
        )

        cov_part2 = (kappa / (phi**4 - phi**2)) * (
            self._kernel(1.0, phi * (lag - 1.0) * timescale, nu, alpha)
            - 2 * self._kernel(1.0, phi * lag * timescale, nu, alpha)
            + self._kernel(1.0, phi * (lag + 1.0) * timescale, nu, alpha)
        )
        return lambda_ * mu_c * iota**2 * (cov_part1 - cov_part2)

    def moment_3rd(self, timescale: float, params: Optional[BLRPRx_params]) -> float:
        lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = self.unpack_params(params)
        mu_c = 1.0 + kappa / phi
        f1 = self.rci_model.get_f1(sigmax_mux=sigmax_mux)
        f2 = self.rci_model.get_f2(sigmax_mux=sigmax_mux)

        phi2 = phi**2
        phi3 = phi**3
        phi4 = phi**4
        phi5 = phi**5
        phi6 = phi**6
        phi7 = phi**7
        phi8 = phi**8
        phi9 = phi**9

        kappa2 = kappa**2

        m3_part0 = (
            (1 + 2 * phi + phi2) * (phi4 - 2 * phi3 - 3 * phi2 + 8 * phi - 4) * phi3
        )
        m3_part1 = self._kernel(1.0, timescale, nu, alpha) * (
            12 * phi7 * kappa2
            - 24 * f1 * phi2 * kappa
            - 18 * phi4 * kappa2
            + 24 * f1 * phi3 * kappa
            - 132 * f1 * phi6 * kappa
            + 150 * f1 * phi4 * kappa
            - 42 * phi5 * kappa2
            - 6 * f1 * phi5 * kappa
            + 108 * phi5 * f2
            - 72 * phi7 * f2
            - 48 * phi3 * f2
            + 24 * f1 * phi8 * kappa
            + 12 * phi3 * kappa2
            + 12 * phi9 * f2
        )

        m3_part2 = self._kernel(0, timescale, nu, alpha) * (
            24 * f1 * phi4 * timescale * kappa
            + 6 * f2 * phi9 * timescale
            - 30 * f1 * phi6 * timescale * kappa
            + 6 * f1 * phi8 * timescale * kappa
            + 54 * f2 * phi5 * timescale
            - 24 * f2 * phi3 * timescale
            - 36 * f2 * phi7 * timescale
        )

        m3_part3 = self._kernel(1.0, phi * timescale, nu, alpha) * (
            -48 * kappa2
            + 6 * f1 * phi4 * kappa
            - 48 * f1 * phi * kappa
            + 6 * phi5 * kappa2
            - 24 * f1 * phi2 * kappa
            + 36 * f1 * phi3 * kappa
            - 6 * f1 * phi5 * kappa
            + 84 * phi2 * kappa2
            + 12 * phi3 * kappa2
            - 18 * phi4 * kappa2
        )

        m3_part4 = self._kernel(0, phi * timescale, nu, alpha) * (
            -24 * phi * timescale * kappa2
            + 30 * phi3 * timescale * kappa2
            - 6 * phi5 * timescale * kappa2
        )

        m3_part5 = self._kernel(1.0, 0, nu, alpha) * (
            +72 * f2 * phi7
            + 48 * f1 * phi * kappa
            + 24 * f1 * phi2 * kappa
            - 36 * f1 * phi3 * kappa
            - 84 * phi2 * kappa2
            + 6 * f1 * phi5 * kappa
            + 117 * f1 * phi6 * kappa
            + 39 * phi5 * kappa2
            - 12 * f2 * phi9
            - 138 * f1 * phi4 * kappa
            + 48 * kappa2
            - 9 * phi7 * kappa2
            + 48 * f2 * phi3
            + 18 * phi4 * kappa2
            - 21 * f1 * phi8 * kappa
            - 12 * phi3 * kappa2
            - 108 * f2 * phi5
        )

        m3_part6 = timescale * (
            -24 * phi * kappa2
            - 72 * f1 * phi6 * kappa
            - 36 * phi5 * kappa2
            + 54 * phi3 * kappa2
            + 6 * phi7 * kappa2
            + 54 * f2 * phi5
            - 36 * f2 * phi7
            - 24 * f2 * phi3
            - 48 * f1 * phi2 * kappa
            + 12 * f1 * phi8 * kappa
            + 6 * f2 * phi9
            + 108 * f1 * phi4 * kappa
        )

        m3_part7 = self._kernel(1.0, 2 * timescale, nu, alpha) * (
            -12 * f1 * phi4 * kappa
            - 3 * f1 * phi8 * kappa
            + 15 * f1 * phi6 * kappa
            - 3 * phi7 * kappa2
            + 3 * phi5 * kappa2
        )

        m3_part8 = self._kernel(1.0, (1 + phi) * timescale, nu, alpha) * (
            -24 * f1 * phi3 * kappa
            - 6 * f1 * phi4 * kappa
            + 6 * f1 * phi5 * kappa
            + 24 * f1 * phi2 * kappa
            + 18 * phi4 * kappa2
            - 12 * phi3 * kappa2
            - 6 * phi5 * kappa2
        )

        return (lambda_ * mu_c * iota**3 / m3_part0) * (
            m3_part1
            + m3_part2
            + m3_part3
            + m3_part4
            + m3_part5
            + m3_part6
            + m3_part7
            + m3_part8
        )

    def prop(
        self,
        prop: BL_Props,
        timescale: float = 1.0,
        params: Optional[BLRPRx_params] = None,
    ) -> float:
        if prop == BL_Props.MEAN:
            return self.mean(timescale, params)
        elif prop == BL_Props.VAR:
            return self.variance(timescale, params)
        elif prop == BL_Props.CVAR:
            return np.sqrt(self.variance(timescale, params)) / self.mean(
                timescale, params
            )
        elif prop == BL_Props.AR1:
            return self.covariance(timescale, 1.0, params) / self.variance(
                timescale, params
            )
        elif prop == BL_Props.AR2:
            return self.covariance(timescale, 2.0, params) / self.variance(
                timescale, params
            )
        elif prop == BL_Props.AR3:
            return self.covariance(timescale, 3.0, params) / self.variance(
                timescale, params
            )
        elif prop == BL_Props.AC1:
            return self.covariance(timescale, 1.0, params)
        elif prop == BL_Props.AC2:
            return self.covariance(timescale, 2.0, params)
        elif prop == BL_Props.AC3:
            return self.covariance(timescale, 3.0, params)
        elif prop == BL_Props.SKEWNESS:
            return self.moment_3rd(timescale, params) / np.power(
                self.variance(timescale, params), 1.5
            )
        else:
            raise ValueError(f"Prop {prop} is not supported in BLRPRx. ")

    def sample(
        self, duration_hr: float, params: Optional[BLRPRx_params] = None
    ) -> npt.NDArray[np.float64]:
        lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = self.unpack_params(params)
        rng = self.rng

        # Calculate the original parameters
        eta = rng.gamma(alpha, 1 / nu)
        gamma = phi * eta
        beta = kappa * eta
        mux = iota * eta

        # Storm sampling
        n_storm = rng.poisson(lambda_ * duration_hr)
        storm_starts = rng.uniform(0, duration_hr, n_storm)
        storm_durations = rng.exponential(1 / gamma, n_storm)

        # Cell sampling
        n_cells_per_storm = 1 + rng.poisson(beta * storm_durations, size=n_storm)
        total_cells: int = n_cells_per_storm.sum()

        # Pre-allocate arrays
        cell_starts = np.zeros(total_cells)
        cells_start_idx = 0
        for i, (s, d) in enumerate(zip(storm_starts, storm_durations)):
            cell_starts[
                cells_start_idx
            ] = s  # First cell starts at the same time as the storm
            cell_starts[
                cells_start_idx + 1 : cells_start_idx + n_cells_per_storm[i]
            ] = rng.uniform(s, s + d, n_cells_per_storm[i] - 1)
            cells_start_idx += n_cells_per_storm[i]

        cell_durations = rng.exponential(scale=1 / eta, size=total_cells)
        cell_intensities = self.rci_model.sample_intensity(
            mux=mux, sigmax_mux=sigmax_mux, n_cells=total_cells
        )
        cell_ends = cell_starts + cell_durations

        # Flatten cell_starts, cell_ends, cell_intensities and stack them together
        cell_arr = np.stack((cell_starts, cell_ends, cell_intensities), axis=-1)

        return cell_arr
