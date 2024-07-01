from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, List, Optional, Tuple, Union

import numba as nb  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import scipy as sp  # type: ignore

from pybl.models import BaseBLRP, BaseBLRP_RCIModel, StatMetrics
from pybl.raincell import ExponentialRCIModel, IConstantRCI, Storm
from pybl.timeseries import IndexedSnapshot


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
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
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
                lambda_=5.0,
                phi=5.0,
                kappa=5.0,
                alpha=5.0,
                nu=5.0,
                sigmax_mux=5.0,
                iota=5.0,
            )
        else:
            self.params = params.copy()


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

    def _kernel(self, k: float, u: float, nu: float, alpha: float) -> float:
        return _blrprx_kernel(k, u, nu, alpha)

    def mean(self, timescale: float = 1.0) -> float:
        return _blrprx_mean(timescale, *self.get_params().unpack())

    def variance(self, timescale: float = 1.0) -> float:
        f1 = self.rci_model.get_f1(sigmax_mux=self.params.sigmax_mux)
        return _blrprx_variance(timescale, f1, *self.get_params().unpack())

    def covariance(self, timescale: float = 1.0, lag: float = 1.0) -> float:
        f1 = self.rci_model.get_f1(sigmax_mux=self.params.sigmax_mux)
        return _blrprx_covariance(timescale, f1, lag, *self.get_params().unpack())

    def moment_3rd(self, timescale: float = 1.0) -> float:
        f1 = self.rci_model.get_f1(sigmax_mux=self.params.sigmax_mux)
        f2 = self.rci_model.get_f2(sigmax_mux=self.params.sigmax_mux)
        return _blrprx_moment_3rd(timescale, f1, f2, *self.get_params().unpack())

    def get_stats(
        self,
        stat_metric: StatMetrics,
        timescale: Union[float, timedelta] = 1.0,
    ) -> float:
        if isinstance(timescale, timedelta):
            timescale = timescale.total_seconds() / 3600.0

        if stat_metric == StatMetrics.MEAN:
            return self.mean(timescale)
        elif stat_metric == StatMetrics.VAR:
            return self.variance(timescale)
        elif stat_metric == StatMetrics.CVAR:
            return np.sqrt(self.variance(timescale)) / self.mean(timescale)
        elif stat_metric == StatMetrics.AR1:
            return self.covariance(timescale, 1.0) / self.variance(timescale)
        elif stat_metric == StatMetrics.AR2:
            return self.covariance(timescale, 2.0) / self.variance(timescale)
        elif stat_metric == StatMetrics.AR3:
            return self.covariance(timescale, 3.0) / self.variance(timescale)
        elif stat_metric == StatMetrics.AC1:
            return self.covariance(timescale, 1.0)
        elif stat_metric == StatMetrics.AC2:
            return self.covariance(timescale, 2.0)
        elif stat_metric == StatMetrics.AC3:
            return self.covariance(timescale, 3.0)
        elif stat_metric == StatMetrics.SKEWNESS:
            return self.moment_3rd(timescale) / np.power(self.variance(timescale), 1.5)
        else:
            # Not implemented properties
            return 0.0
    def get_stats_dataframe(
        self,
        stat_metrics: list[StatMetrics],
        timescales: Union[list[float], list[timedelta]] = [1.0],
    ) -> pd.DataFrame:
        for scale_idx, scale in enumerate(timescales):
            if not (isinstance(scale, float) or isinstance(scale, timedelta)):
                raise ValueError("timescales must be a list of float or timedelta")

        stats_arr = np.empty((len(timescales), len(stat_metrics)), dtype=np.float64)
        for scale_idx, scale in enumerate(timescales):

            if isinstance(scale, timedelta):
                float_scale = scale.total_seconds() / 3600.0
            elif isinstance(scale, float):
                float_scale = scale

            for prop_idx, prop in enumerate(stat_metrics):
                stats_arr[scale_idx, prop_idx] = self.get_stats(prop, float_scale)

        stats_df = pd.DataFrame(
            stats_arr, columns=stat_metrics, index=timescales
        )
        stats_df.index.name = "timescale_hr"

        return stats_df

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
            cell_starts[cells_start_idx] = (
                s  # First cell starts at the same time as the storm
            )
            cell_starts[
                cells_start_idx + 1 : cells_start_idx + n_cells_per_storm[i]
            ] = rng.uniform(s, s + d, n_cells_per_storm[i] - 1)

            cell_durations[cells_start_idx : cells_start_idx + n_cells_per_storm[i]] = (
                rng.exponential(scale=1 / eta[i], size=n_cells_per_storm[i])
            )

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

    def sample(self, duration_hr: float) -> IndexedSnapshot:
        # cell_arr = self.sample_raw(duration_hr)
        cell_arr, n_cells_per_storm, storms_info = _blrprx_sample(
            duration_hr,
            self.rci_model.sample_intensity,
            self.rng,
            *self.get_params().unpack(),
        )

        delta = np.concatenate(
            [
                np.stack([cell_arr[:, 0], cell_arr[:, 2]]).T,
                np.stack([cell_arr[:, 1], -cell_arr[:, 2]]).T,
            ],
            axis=0,
        )
        ts = IndexedSnapshot.fromDelta(time=delta[:, 0], intensity_delta=delta[:, 1])

        return ts[0:duration_hr]  # type: ignore

    def sample_storms(self, duration_hr: float) -> Tuple[IndexedSnapshot, List[Storm]]:
        cell_arr, n_cells_per_storm, storms_info = _blrprx_sample(
            duration_hr,
            self.rci_model.sample_intensity,
            self.rng,
            *self.get_params().unpack(),
        )

        delta = np.concatenate(
            [
                np.stack([cell_arr[:, 0], cell_arr[:, 2]]).T,
                np.stack([cell_arr[:, 1], -cell_arr[:, 2]]).T,
            ],
            axis=0,
        )
        ts = IndexedSnapshot.fromDelta(time=delta[:, 0], intensity_delta=delta[:, 1])

        storms = []

        cum_cells = 0
        for i, storm_info in enumerate(storms_info):
            storms.append(
                Storm(
                    cells=cell_arr[cum_cells : cum_cells + n_cells_per_storm[i]],
                    start=storm_info[0],
                    duration=storm_info[1],
                    eta=storm_info[2],
                    mux=storm_info[3],
                    gamma=storm_info[4],
                    beta=storm_info[5],
                )
            )
            cum_cells += n_cells_per_storm[i]

        return ts[0:duration_hr], storms  # type: ignore

    def fit(
        self,
        stats: pd.DataFrame,
        weight: pd.DataFrame,
        rng: Optional[np.random.Generator] = None,
        tol: float = 0.5,
    ) -> dict[str, Any]:

        warnings.filterwarnings("ignore", category=sp.optimize.OptimizeWarning)


        if rng is None:
            rng = np.random.default_rng()
        bound = self.bound_estimation(stats, weight)

        fitter = BLRPRxConfig(stats, weight, self.rci_model)
        obj = fitter.get_evaluation_func()

        for i in range(20):
            if obj(np.array(self.params.unpack())) < tol:
                break

            # Clip the parameters to the bound

            guess = np.clip(np.array(self.params.unpack()), [b[0] for b in bound], [b[1] for b in bound])

            result_bh = sp.optimize.basinhopping(
                obj,
                x0=guess,
                T=10,
                stepsize=1,
                niter=20,
                minimizer_kwargs={"method": "Nelder-Mead", "bounds": bound},
                seed=rng,
            )

            if obj(np.array(self.params.unpack())) > result_bh.fun:
                self.update_params(BLRPRx_params(*result_bh.x))

        if obj(np.array(self.params.unpack())) < tol:
            status = "Success"
        else:
            status = "Maximum iteration reached"

        stats_metrics: list[StatMetrics] = stats.columns.to_list() # type: ignore
        timescales = stats.index.to_list()

        report = {
            "obj": obj,
            "fun": obj(np.array(self.params.unpack())),
            "x": self.params.unpack(),
            "status": status,
            "rci_model": self.rci_model.__class__.__name__,
            "theo_stats": self.get_stats_dataframe(stat_metrics=stats_metrics, timescales=timescales),
        }

        return report
        # if obj(np.array(self.params.unpack())) < result_bh.fun:
        #    self.update_params(BLRPRx_params(*result_bh.x))

    def bound_estimation(
        self, target: pd.DataFrame, weight: pd.DataFrame
    ) -> List[Tuple[float, float]]:
        bound = [
            (0.0001, 10.0),
            (0.0001, 10.0),
            (0.0001, 10.0),
            (0.0001, 10.0),
            (0.0001, 10.0),
            (0.999, 1.0),
            (0.0001, 10.0),
        ]
        return bound


class BLRPRxConfig:
    __slots__ = ("_weight", "_stats", "_rci_model")

    _weight: pd.DataFrame
    _stats: pd.DataFrame
    _rci_model: IConstantRCI

    def __init__(
        self,
        stats: pd.DataFrame,
        weight: pd.DataFrame,
        rci_model: Optional[IConstantRCI] = None,
    ):
        if not (stats.shape == weight.shape):
            raise ValueError("stats and weight must have the same shape")

        # Check if they have the same set of column regardless of order
        if not (set(stats.columns) == set(weight.columns)):
            raise ValueError("stats and weight must have the same set of columns")

        # Check if they have the same set of index regardless of order
        if not (set(stats.index) == set(weight.index)):
            raise ValueError("stats and weight must have the same set of index")

        # Make sure there isn't duplicate columns
        if len(stats.columns) != len(set(stats.columns)) or len(weight.columns) != len(
            set(weight.columns)
        ):
            raise ValueError("stats and weight must not have duplicate columns")

        # Make sure there isn't duplicate index
        if len(stats.index) != len(set(stats.index)) or len(weight.index) != len(
            set(weight.index)
        ):
            raise ValueError("stats and weight must not have duplicate index")

        # Check if rci_model is None or IConstantRCI
        if rci_model is None:
            self._rci_model = ExponentialRCIModel()
        elif isinstance(rci_model, IConstantRCI):
            self._rci_model = rci_model
        else:
            raise ValueError("rci_model must follow IConstantRCI protocol")

        self._stats = stats.copy(deep=True)
        self._weight = weight.copy(deep=True)

    def get_evaluation_func(self) -> Callable[[npt.NDArray[np.float64]], np.float64]:
        """
        Return a objective function based on your configuration of stats, weight and mask.
        It is a function that takes in a numpy array of 7 parameters and return a score.

        Note that the first time this function is called, it will take a while to compile.

        Returns:
        --------
        evaluation_func: Callable[[npt.NDArray[np.float64]], np.float64]
            A function that takes in a numpy array of 7 parameters and return a error score.
        """
        scales = self._stats.index.to_numpy()
        stats_types_enum = self._stats.columns.values
        for i, stat in enumerate(stats_types_enum):
            if stat not in StatMetrics:
                raise ValueError(f"Invalid StatMetrics: {stat}")
        stats_types = np.array([stat.value for stat in stats_types_enum])

        stats_np = self._stats.to_numpy()
        weight_np = self._weight.to_numpy()

        f1_func = self._rci_model.get_f1
        f2_func = self._rci_model.get_f2

        @nb.njit
        def evaluation_func(x: npt.NDArray[np.float64]) -> np.float64:
            # fmt: off
            (_l ,_p ,_k ,_a ,_n ,_s ,_i) = x  # noqa: E741.   lambda, phi, kappa, alpha, nu, sigma, iota

            # If phi is 1. Many of the formula will have division by 0.
            if _p == 1.0:
                return np.float64(np.nan)

            # fmt: on
            f1 = f1_func(_s)
            f2 = f2_func(_s)
            score = np.float64(0)
            for t, scale in enumerate(scales):
                mean = _blrprx_mean(scale, _l, _p, _k, _a, _n, _s, _i)
                variance = _blrprx_variance(scale, f1, _l, _p, _k, _a, _n, _s, _i)
                if variance < 0:
                    return np.float64(np.nan)
                moment_3rd = _blrprx_moment_3rd(
                    scale, f1, f2, _l, _p, _k, _a, _n, _s, _i
                )

                for stat_idx, stat in enumerate(stats_types):
                    if weight_np[t, stat_idx] == 0:
                        continue
                    predict = 0
                    if stat == 0:  # Stat_Props.MEAN.value
                        predict = mean
                    elif stat == 1:  # Stat_Props.CVAR.value
                        predict = np.sqrt(variance) / mean
                    elif stat == 2:  # Stat_Props.SKEWNESS.value
                        predict = moment_3rd / np.power(variance, 1.5)
                    elif stat == 3:  # Stat_Props.AR1.value
                        predict = (
                            _blrprx_covariance(
                                scale, f1, 1.0, _l, _p, _k, _a, _n, _s, _i
                            )
                            / variance
                        )
                    elif stat == 4:  # Stat_Props.AR2.value
                        predict = (
                            _blrprx_covariance(
                                scale, f1, 2.0, _l, _p, _k, _a, _n, _s, _i
                            )
                            / variance
                        )
                    elif stat == 5:  # Stat_Props.AR3.value
                        predict = (
                            _blrprx_covariance(
                                scale, f1, 3.0, _l, _p, _k, _a, _n, _s, _i
                            )
                            / variance
                        )
                    elif stat == 6:  # Stat_Props.pDRY.value
                        predict = 0
                    elif stat == 7:  # Stat_Props.MSIT.value
                        predict = 0
                    elif stat == 8:  # Stat_Props.MSD.value
                        predict = 0
                    elif stat == 9:  # Stat_Props.MCIT.value
                        predict = 0
                    elif stat == 10:  # Stat_Props.MCD.value
                        predict = 0
                    elif stat == 11:  # Stat_Props.MCS.value
                        predict = 0
                    elif stat == 12:  # Stat_Props.MPC.value
                        predict = 0
                    elif stat == 13:  # Stat_Props.VAR.value
                        predict = variance
                    elif stat == 14:  # Stat_Props.AC1.value
                        predict = _blrprx_covariance(
                            scale, f1, 1.0, _l, _p, _k, _a, _n, _s, _i
                        )
                    elif stat == 15:  # Stat_Props.AC2.value
                        predict = _blrprx_covariance(
                            scale, f1, 2.0, _l, _p, _k, _a, _n, _s, _i
                        )
                    elif stat == 16:  # Stat_Props.AC3.value
                        predict = _blrprx_covariance(
                            scale, f1, 3.0, _l, _p, _k, _a, _n, _s, _i
                        )
                    score += (
                        np.power(predict - stats_np[t, stat_idx], 2)
                        * weight_np[t, stat_idx]
                    )
            # return np.nansum(np.power(predict - stats_np, 2) * weight_np * mask_np)
            return score

        return evaluation_func

    @staticmethod
    def default_stats(timescale: List[float] = [1, 3, 6, 24]) -> pd.DataFrame:
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        stats_df = pd.DataFrame(
            columns=[prop for prop in StatMetrics], index=timescale, dtype=np.float64
        )
        stats_df.index.name = "timescales"

        return stats_df

    @staticmethod
    def default_weight(timescale: List[float] = [1, 3, 6, 24]) -> pd.DataFrame:
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24
        weight_df = pd.DataFrame(
            columns=[prop for prop in StatMetrics], index=timescale, dtype=np.float64
        )
        weight_df.index.name = "timescales"
        return weight_df

    @staticmethod
    def default_mask(timescale: List[float] = [1, 3, 6, 24]) -> pd.DataFrame:
        # Create a numpy array with Stat_Props as columns one row with timescales=1,3,6,24, with default value of False
        mask_np = np.zeros((len(timescale), len(StatMetrics)), dtype=np.int_)
        # Create a pandas dataframe with Stat_Props as columns one row with timescales=1,3,6,24, with default value of False
        mask_df = pd.DataFrame(
            data=mask_np,
            columns=[prop for prop in StatMetrics],
            index=timescale,
            dtype=np.int_,
        )
        mask_df.index.name = "timescales"
        return mask_df


@nb.njit()
def _blrprx_kernel(k: float, u: float, nu: float, alpha: float) -> float:
    """
    k: lag
    u: timescale
    """
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
    f1: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
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
    f1: float,
    lag: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
):
    mu_c = 1.0 + kappa / phi

    # TODO: In the original code there's a variable k = 3.0 - 2.0 * c.
    # Is it the same as # Kaczmarska et al. (2014) A.10?
    # The lag between covariance? Seems it doesn't.
    # Because in Kaczmarska et al. (2014) (1) There is also a k. That is not lag right?

    cov_part1 = (f1 + (kappa * phi) / (phi**2 - 1)) * (
        _blrprx_kernel(1.0, (lag - 1.0) * timescale, nu, alpha)
        - 2 * _blrprx_kernel(1.0, lag * timescale, nu, alpha)
        + _blrprx_kernel(1.0, (lag + 1.0) * timescale, nu, alpha)
    )

    cov_part2 = (kappa / (phi**4 - phi**2)) * (
        _blrprx_kernel(1.0, phi * (lag - 1.0) * timescale, nu, alpha)
        - 2 * _blrprx_kernel(1.0, phi * lag * timescale, nu, alpha)
        + _blrprx_kernel(1.0, phi * (lag + 1.0) * timescale, nu, alpha)
    )
    return lambda_ * mu_c * iota**2 * (cov_part1 - cov_part2)


@nb.njit()
def _blrprx_moment_3rd(
    timescale: float,
    f1: float,
    f2: float,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
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
    duration_hr: float,
    intensity_sampler: Callable[
        [np.random.Generator, float, float, int], npt.NDArray[np.float64]
    ],
    rng: np.random.Generator,
    lambda_: float,
    phi: float,
    kappa: float,
    alpha: float,
    nu: float,
    sigmax_mux: float,
    iota: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    # Storm number sampling
    n_storm = rng.poisson(lambda_ * duration_hr)

    total_cells: int = 0
    eta = np.empty(n_storm, dtype=np.float64)
    mux = np.empty(n_storm, dtype=np.float64)
    storm_starts = np.empty(n_storm, dtype=np.float64)
    storm_durations = np.empty(n_storm, dtype=np.float64)
    storm_info = np.empty((n_storm, 6), dtype=np.float64)
    n_cells_per_storm = np.empty(n_storm, dtype=np.int64)
    for i in range(n_storm):
        eta[i] = rng.gamma(alpha, 1 / nu)
        gamma = phi * eta[i]
        beta = kappa * eta[i]
        mux[i] = iota * eta[i]  # mux = iota * eta
        storm_starts[i] = rng.uniform(
            0, duration_hr
        )  # storm_starts = rng.uniform(0, duration_hr, n_storm)
        storm_durations[i] = rng.exponential(
            1 / gamma
        )  # storm_durations = rng.exponential(1 / gamma, n_storm)
        n_cells_per_storm[i] = 1 + rng.poisson(
            beta * storm_durations[i]
        )  # n_cells_per_storm = 1 + rng.poisson(beta * storm_durations, size=n_storm)
        total_cells += n_cells_per_storm[i]
        storm_info[i] = (
            storm_starts[i],
            storm_durations[i],
            eta[i],
            mux[i],
            gamma,
            beta,
        )

    # Pre-allocate arrays
    cell_starts = np.empty(total_cells)  # (total_cells, )
    cell_durations = np.empty(total_cells)  # (total_cells, )
    cell_intensities = np.empty(total_cells)  # (total_cells, )

    cells_start_idx = 0
    for i, (s, d) in enumerate(zip(storm_starts, storm_durations)):
        cell_starts[cells_start_idx] = (
            s  # First cell starts at the same time as the storm
        )
        cell_starts[cells_start_idx + 1 : cells_start_idx + n_cells_per_storm[i]] = (
            rng.uniform(s, s + d, n_cells_per_storm[i] - 1)
        )

        cell_durations[cells_start_idx : cells_start_idx + n_cells_per_storm[i]] = (
            rng.exponential(scale=1 / eta[i], size=n_cells_per_storm[i])
        )

        cell_intensities[cells_start_idx : cells_start_idx + n_cells_per_storm[i]] = (
            intensity_sampler(rng, mux[i], sigmax_mux, n_cells_per_storm[i])
        )
        cells_start_idx += n_cells_per_storm[i]

    cell_ends = cell_starts + cell_durations  # (total_cells, )

    # Flatten cell_starts, cell_ends, cell_intensities and stack them together
    cell_arr = np.stack(
        (cell_starts, cell_ends, cell_intensities), axis=-1
    )  # (total_cells, 3)

    return cell_arr, n_cells_per_storm, storm_info
