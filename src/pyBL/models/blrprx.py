from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore
import math

from pyBL.models import BaseBLRP, BaseBLRP_RCIModel, Stat_Props
from pyBL.timeseries import IntensityMRLE
import numba as nb # type: ignore


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

    def size(self):
        return 7

    def unpack(
        self,
    ) -> Tuple[float, float, float, float, float, float, float,]:
        return (
            self.lambda_,
            self.phi,
            self.kappa,
            self.alpha,
            self.nu,
            self.sigmax_mux,
            self.iota,
        )

    def copy(self) -> BLRPRx_params:
        return BLRPRx_params(*self.unpack())


class BLRPRx(BaseBLRP):
    ## This BLRPRx is an implementation of Kaczmarska et al. (2014)
    __slots__ = (
        "rci_model",
        "rng",
        "params",
        "cache_mean",
        "cache_variance",
        "cache_covariance",
        "cache_moment_3rd",
    )

    def __init__(
        self,
        params: Optional[BLRPRx_params] = None,
        sampling_rng: Optional[np.random.Generator] = None,
        rci_model: BaseBLRP_RCIModel = None,
    ) -> None:
        super().__init__(sampling_rng, rci_model)
        # If user does not provide params, give the default values.
        if params is None:
            self.params = BLRPRx_params(
                lambda_=0.016679,
                phi=0.082,
                kappa=0.349,
                alpha=9.01,
                nu=10,
                sigmax_mux=1,
                iota=0.97,
            )
        else:
            self.params = params.copy()

        self.cache_mean = {}
        self.cache_variance = {}
        self.cache_covariance = {}
        self.cache_moment_3rd = {}

    def copy(self, rng: Optional[np.random.Generator] = None) -> BLRPRx:
        if rng is None:
            rng = np.random.default_rng(self.rng)

        return BLRPRx(
            params=self.get_params(),
            sampling_rng=rng,
            rci_model=type(self.rci_model)(),
        )

    def get_params(self) -> BLRPRx_params:
        return self.params.copy()

    def update_params(self, params: BLRPRx_params) -> None:
        self.params = params.copy()
        self.cache_mean = {}
        self.cache_variance = {}
        self.cache_covariance = {}
        self.cache_moment_3rd = {}

    def _kernel(self, k: float, u: float, nu: float, alpha: float) -> float:
        return _blrprx_kernel(k, u, nu, alpha)

    def mean(self, timescale: float = 1.0) -> float:
        if timescale not in self.cache_mean:
            self.cache_mean[timescale] = _blrprx_mean(
                timescale, *self.get_params().unpack()
            )
        return self.cache_mean[timescale]

    def variance(self, timescale: float = 1.0) -> float:
        f1 = self.rci_model.get_f1(sigmax_mux=self.params.sigmax_mux)
        if timescale not in self.cache_variance:
            self.cache_variance[timescale] = _blrprx_variance(
                timescale,
                *self.get_params().unpack(),
                f1,
            )
        return self.cache_variance[timescale]

    def covariance(self, timescale: float = 1.0, lag: float = 1.0) -> float:
        f1 = self.rci_model.get_f1(sigmax_mux=self.params.sigmax_mux)
        if timescale not in self.cache_covariance:
            self.cache_covariance[timescale] = _blrprx_covariance(
                timescale,
                *self.get_params().unpack(),
                f1,
                lag,
            )
        return self.cache_covariance[timescale]

    def moment_3rd(self, timescale: float = 1.0) -> float:
        f1 = self.rci_model.get_f1(sigmax_mux=self.params.sigmax_mux)
        f2 = self.rci_model.get_f2(sigmax_mux=self.params.sigmax_mux)
        if timescale not in self.cache_moment_3rd:
            self.cache_moment_3rd[timescale] = _blrprx_moment_3rd(
                timescale,
                *self.get_params().unpack(),
                f1,
                f2,
            )
        return self.cache_moment_3rd[timescale]

    def get_prop(
        self,
        prop: Stat_Props,
        timescale: float = 1.0,
    ) -> float:
        if prop == Stat_Props.MEAN:
            return self.mean(timescale)
        elif prop == Stat_Props.VAR:
            return self.variance(timescale)
        elif prop == Stat_Props.CVAR:
            return np.sqrt(self.variance(timescale)) / self.mean(timescale)
        elif prop == Stat_Props.AR1:
            return self.covariance(timescale, 1.0) / self.variance(timescale)
        elif prop == Stat_Props.AR2:
            return self.covariance(timescale, 2.0) / self.variance(timescale)
        elif prop == Stat_Props.AR3:
            return self.covariance(timescale, 3.0) / self.variance(timescale)
        elif prop == Stat_Props.AC1:
            return self.covariance(timescale, 1.0)
        elif prop == Stat_Props.AC2:
            return self.covariance(timescale, 2.0)
        elif prop == Stat_Props.AC3:
            return self.covariance(timescale, 3.0)
        elif prop == Stat_Props.SKEWNESS:
            return self.moment_3rd(timescale) / np.power(self.variance(timescale), 1.5)
        else:
            # Not implemented properties
            return 0.0

    def sample_raw(self, duration_hr: float) -> npt.NDArray[np.float64]:
        lambda_, phi, kappa, alpha, nu, sigmax_mux, iota = self.get_params().unpack()
        rng = self.rng

        # Storm number sampling
        n_storm = rng.poisson(lambda_ * duration_hr)

        # Calculate the original parameters
        eta = rng.gamma(alpha, 1 / nu, size=n_storm)  # (n_storm, )
        gamma = phi * eta  # (n_storm, )
        beta = kappa * eta  # (n_storm, )
        mux = iota * eta  # (n_storm, )

        # Storm parameters sampling
        storm_starts = rng.uniform(0, duration_hr, n_storm)  # (n_storm, )
        storm_durations = rng.exponential(1 / gamma, n_storm)  # (n_storm, )

        # Cell sampling
        n_cells_per_storm = 1 + rng.poisson(
            beta * storm_durations, size=n_storm
        )  # (n_storm, )
        total_cells: int = n_cells_per_storm.sum()

        # Pre-allocate arrays
        cell_starts = np.zeros(total_cells)  # (total_cells, )
        cell_durations = np.zeros(total_cells)  # (total_cells, )
        cell_intensities = np.zeros(total_cells)  # (total_cells, )

        cells_start_idx = 0
        for i, (s, d) in enumerate(zip(storm_starts, storm_durations)):
            cell_starts[
                cells_start_idx
            ] = s  # First cell starts at the same time as the storm
            cell_starts[
                cells_start_idx + 1 : cells_start_idx + n_cells_per_storm[i]
            ] = rng.uniform(s, s + d, n_cells_per_storm[i] - 1)

            cell_durations[
                cells_start_idx : cells_start_idx + n_cells_per_storm[i]
            ] = rng.exponential(scale=1 / eta[i], size=n_cells_per_storm[i])

            cell_intensities[
                cells_start_idx : cells_start_idx + n_cells_per_storm[i]
            ] = self.rci_model.sample_intensity(
                rng=rng, mux=mux[i], sigmax_mux=sigmax_mux, n_cells=n_cells_per_storm[i]
            )
            cells_start_idx += n_cells_per_storm[i]

        cell_ends = cell_starts + cell_durations  # (total_cells, )

        # Flatten cell_starts, cell_ends, cell_intensities and stack them together
        cell_arr = np.stack(
            (cell_starts, cell_ends, cell_intensities), axis=-1
        )  # (total_cells, 3)

        return cell_arr

    def sample(self, duration_hr: float) -> IntensityMRLE:
        #cell_arr = self.sample_raw(duration_hr)
        cell_arr = _blrprx_sample(*self.get_params().unpack(), duration_hr, self.rci_model.sample_intensity, self.rng)
        delta = np.concatenate(
            [
                np.stack([cell_arr[:, 0], cell_arr[:, 2]]).T,
                np.stack([cell_arr[:, 1], -cell_arr[:, 2]]).T,
            ],
            axis=0,
        )
        return IntensityMRLE.fromDelta(time=delta[:, 0], intensity_delta=delta[:, 1])

@nb.njit()
def _blrprx_kernel(k: float, u: float, nu: float, alpha: float) -> float:
    '''
    k: lag
    u: timescale
    '''
    # Modelling rainfall with a Bartlett–Lewis process: new developments(2020) Formula (5)

    ## TODO: Check if this is still required.
    # if alpha <= 4.0 and np.modf(alpha)[0] == 0.0:
    #    alpha += 1.0e-8

    ## TODO: Check if this is still required.
    # if alpha - k >= 171.0 or alpha >= 171.0:
    #    return np.inf

    return (
        np.power(nu / (nu + u), alpha)
        * np.power(nu + u, k)
        * math.gamma(alpha - k)
        / math.gamma(alpha)
    )


@nb.njit()
def _blrprx_mean(
    timescale: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
):
    mu_c = 1.0 + kappa / phi
    # TODO: Check the formula. Why do we have a kernel here?
    return timescale * lambda_ * iota * mu_c * _blrprx_kernel(0, 0, nu, alpha)


@nb.njit()
def _blrprx_variance(
    timescale: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
    f1: float,
):
    mu_c = 1.0 + kappa / phi

    # TODO: The formula is not the same as C/C++ code. Check this.
    # The formula is the same as Kaczmarska et al. (2014) A.2
    var_part1 = timescale * (f1 + kappa / phi)
    var_part2 = (kappa * (1 - phi**3) / (phi**4 - phi**2) - f1) * _blrprx_kernel(
        1.0, 0, nu, alpha
    )
    var_part3 = (kappa / (phi**4 - phi**2)) * _blrprx_kernel(
        1.0, phi * timescale, nu, alpha
    )
    var_part4 = (f1 + kappa * phi / (phi**2 - 1)) * _blrprx_kernel(
        1.0, timescale, nu, alpha
    )

    return (
        2 * lambda_ * mu_c * iota**2 * (var_part1 + var_part2 - var_part3 + var_part4)
    )


@nb.njit()
def _blrprx_covariance(
    timescale: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
    f1: float,
    k: float,
):
    mu_c = 1.0 + kappa / phi

    # TODO: In the original code there's a variable k = 3.0 - 2.0 * c.
    # Is it the same as # Kaczmarska et al. (2014) A.10?
    # The lag between covariance? Seems it doesn't.
    # Because in Kaczmarska et al. (2014) (1) There is also a k. That is not lag right?

    cov_part1 = (f1 + (kappa * phi) / (phi**2 - 1)) * (
        _blrprx_kernel(1.0, (k - 1.0) * timescale, nu, alpha)
        - 2 * _blrprx_kernel(1.0, k * timescale, nu, alpha)
        + _blrprx_kernel(1.0, (k + 1.0) * timescale, nu, alpha)
    )

    cov_part2 = (kappa / (phi**4 - phi**2)) * (
        _blrprx_kernel(1.0, phi * (k - 1.0) * timescale, nu, alpha)
        - 2 * _blrprx_kernel(1.0, phi * k * timescale, nu, alpha)
        + _blrprx_kernel(1.0, phi * (k + 1.0) * timescale, nu, alpha)
    )
    return lambda_ * mu_c * iota**2 * (cov_part1 - cov_part2)


@nb.njit()
def _blrprx_moment_3rd(
    timescale: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
    f1: float,
    f2: float,
):
    mu_c = 1.0 + kappa / phi

    phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9 = np.power(phi, np.arange(2, 10))

    kappa2 = kappa**2

    m3_part0 = (1 + 2 * phi + phi2) * (phi4 - 2 * phi3 - 3 * phi2 + 8 * phi - 4) * phi3
    m3_part1 = _blrprx_kernel(1.0, timescale, nu, alpha) * (
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

    m3_part2 = _blrprx_kernel(0, timescale, nu, alpha) * (
        24 * f1 * phi4 * timescale * kappa
        + 6 * f2 * phi9 * timescale
        - 30 * f1 * phi6 * timescale * kappa
        + 6 * f1 * phi8 * timescale * kappa
        + 54 * f2 * phi5 * timescale
        - 24 * f2 * phi3 * timescale
        - 36 * f2 * phi7 * timescale
    )

    m3_part3 = _blrprx_kernel(1.0, phi * timescale, nu, alpha) * (
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

    m3_part4 = _blrprx_kernel(0, phi * timescale, nu, alpha) * (
        -24 * phi * timescale * kappa2
        + 30 * phi3 * timescale * kappa2
        - 6 * phi5 * timescale * kappa2
    )

    m3_part5 = _blrprx_kernel(1.0, 0, nu, alpha) * (
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

    m3_part7 = _blrprx_kernel(1.0, 2 * timescale, nu, alpha) * (
        -12 * f1 * phi4 * kappa
        - 3 * f1 * phi8 * kappa
        + 15 * f1 * phi6 * kappa
        - 3 * phi7 * kappa2
        + 3 * phi5 * kappa2
    )

    m3_part8 = _blrprx_kernel(1.0, (1 + phi) * timescale, nu, alpha) * (
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

@nb.njit()
def _blrprx_sample(
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
    duration_hr: float,
    intensity_sampler: Callable[[np.random.Generator, float, float, int], npt.NDArray[np.float64]],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    # Storm number sampling
    n_storm = rng.poisson(lambda_ * duration_hr)

    total_cell: int = 0
    eta = np.empty(n_storm, dtype=np.float64)
    mux = np.empty(n_storm, dtype=np.float64)
    storm_starts = np.empty(n_storm, dtype=np.float64)
    storm_durations = np.empty(n_storm, dtype=np.float64)
    n_cells_per_storm = np.empty(n_storm, dtype=np.int64)
    for i in range(n_storm):
        eta[i] = rng.gamma(alpha, 1 / nu)
        gamma = phi * eta[i]
        beta = kappa * eta[i]
        mux[i] = iota * eta[i] # mux = iota * eta
        storm_starts[i] = rng.uniform(0, duration_hr) # storm_starts = rng.uniform(0, duration_hr, n_storm)
        storm_durations[i] = rng.exponential(1 / gamma) # storm_durations = rng.exponential(1 / gamma, n_storm)
        n_cells_per_storm[i] = 1 + rng.poisson(beta * storm_durations[i]) # n_cells_per_storm = 1 + rng.poisson(beta * storm_durations, size=n_storm)
        total_cell += n_cells_per_storm[i]
    
    total_cells: int = n_cells_per_storm.sum()

    # Pre-allocate arrays
    cell_starts = np.empty(total_cells)  # (total_cells, )
    cell_durations = np.empty(total_cells)  # (total_cells, )
    cell_intensities = np.empty(total_cells)  # (total_cells, )

    cells_start_idx = 0
    for i, (s, d) in enumerate(zip(storm_starts, storm_durations)):
        cell_starts[
            cells_start_idx
        ] = s  # First cell starts at the same time as the storm
        cell_starts[
            cells_start_idx + 1 : cells_start_idx + n_cells_per_storm[i]
        ] = rng.uniform(s, s + d, n_cells_per_storm[i] - 1)

        cell_durations[
            cells_start_idx : cells_start_idx + n_cells_per_storm[i]
        ] = rng.exponential(scale=1 / eta[i], size=n_cells_per_storm[i])

        cell_intensities[
            cells_start_idx : cells_start_idx + n_cells_per_storm[i]
        ] = intensity_sampler(
            rng=rng, mux=mux[i], sigmax_mux=sigmax_mux, n_cells=n_cells_per_storm[i]
        )
        cells_start_idx += n_cells_per_storm[i]

    cell_ends = cell_starts + cell_durations  # (total_cells, )

    # Flatten cell_starts, cell_ends, cell_intensities and stack them together
    cell_arr = np.stack(
        (cell_starts, cell_ends, cell_intensities), axis=-1
    )  # (total_cells, 3)

    return cell_arr