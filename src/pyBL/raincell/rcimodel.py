from __future__ import annotations

from typing import Optional, Protocol, Union, overload, runtime_checkable

import numpy as np
import numpy.typing as npt
import numba as nb  # type: ignore
import scipy as sp  # type: ignore


@runtime_checkable
class IConstantRCI(Protocol):
    @staticmethod
    def get_f1(sigmax_mux: float) -> float:
        ...

    @staticmethod
    def get_f2(sigmax_mux: float) -> float:
        ...

    @staticmethod
    def sample_intensity(
        rng: np.random.Generator, mux: float, sigmax_mux: float, n_cells: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        ...


class GammaRCIModel(IConstantRCI):
    @staticmethod
    @nb.njit
    def get_f1(sigmax_mux: float) -> float:
        return (sigmax_mux + 1.0) / sigmax_mux

    @staticmethod
    @nb.njit
    def get_f2(sigmax_mux: float) -> float:
        x2 = sigmax_mux**2
        return (x2 + 3.0 * sigmax_mux + 2.0) / x2

    @staticmethod
    def sample_intensity(
        rng: np.random.Generator, mux: float, sigmax_mux: float, n_cells: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        if n_cells is None:
            n_cells = 1
        return rng.gamma(sigmax_mux, mux / sigmax_mux, size=n_cells)


class ExponentialRCIModel(IConstantRCI):
    @staticmethod
    @nb.njit
    def get_f1(sigmax_mux: float) -> float:
        return 2.0

    @staticmethod
    @nb.njit
    def get_f2(sigmax_mux: float) -> float:
        return 6.0

    @staticmethod
    def sample_intensity(
        rng: np.random.Generator, mux: float, sigmax_mux: float, n_cells: Optional[int]=None
    ) -> npt.NDArray[np.float64]:
        I_shape = 1.0 / sigmax_mux**2
        I_scale = sigmax_mux**2 * mux
        # TODO: Check this c++ code and Legacy pyBL code
        if n_cells is None or n_cells == 1:
            n_cells = 1

        return rng.gamma(I_shape, I_scale, size=n_cells)

class WeibullRCIModel(IConstantRCI):
    @staticmethod
    @nb.njit
    def get_f1(sigmax_mux: float) -> float:
        ex2: float = sp.special.gamma(1.0 + 2.0 / sigmax_mux)
        ex: float = sp.special.gamma(1.0 + 1.0 / sigmax_mux)
        return ex2 / ex**2

    @staticmethod
    @nb.njit
    def get_f2(sigmax_mux: float) -> float:
        ex3: float = sp.special.gamma(1.0 + 3.0 / sigmax_mux)
        ex: float = sp.special.gamma(1.0 + 1.0 / sigmax_mux)
        return ex3 / ex**3

    @staticmethod
    def sample_intensity(
        rng: np.random.Generator, mux: float, sigmax_mux: float, n_cells: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        I_scale = mux / sp.special.gamma(1.0 + 1.0 / sigmax_mux)
        if n_cells is None:
            n_cells = 1

        # The weibull distribution in numpy differs from the one in c++
        # See Numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.weibull.html
        # See GSL: https://www.gnu.org/software/gsl/doc/html/randist.html#c.gsl_ran_weibull
        return rng.weibull(sigmax_mux, size=n_cells) * I_scale

class ParetoRCIModel(IConstantRCI):
    @staticmethod
    @nb.njit
    def get_f1(sigmax_mux: float) -> float:
        return (sigmax_mux - 1.0) ** 2 / sigmax_mux / (sigmax_mux - 2.0)

    @staticmethod
    @nb.njit
    def get_f2(sigmax_mux: float) -> float:
        return (sigmax_mux - 1.0) ** 3 / sigmax_mux**2 / (sigmax_mux - 3.0)

    @staticmethod
    def sample_intensity(
        rng: np.random.Generator, mux: float, sigmax_mux: float, n_cells: Optional[int] = None
    ) -> float:
        I_scale = mux * (sigmax_mux - 1.0) / sigmax_mux
        if n_cells is None:
            n_cells = 1
        # The pareto distribution in numpy differs from the one in c++
        # See Numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.pareto.html
        # See GSL: https://www.gnu.org/software/gsl/doc/html/randist.html#c.gsl_ran_pareto
        return rng.pareto(sigmax_mux, size=n_cells) * I_scale + I_scale


class GPRCIModel(IConstantRCI):
    @staticmethod
    @nb.njit
    def get_f1(sigmax_mux: float) -> float:
        return 2.0 * (1.0 - sigmax_mux) / (1.0 * 2.0 * sigmax_mux)

    @staticmethod
    @nb.njit
    def get_f2(sigmax_mux: float) -> float:
        return (
            6.0
            * (sigmax_mux - 1.0) ** 2
            / (1.0 - 3.0 * sigmax_mux)
            / (1.0 - 2.0 * sigmax_mux)
        )

    def sample_intensity(
        self, start: float, end: float, mux: float, sigmax_mux: float
    ) -> float:
        I_scale = mux * (1.0 - sigmax_mux)
        I_location = 0.0

        u = 1.0 - self.rng.uniform()

        if sigmax_mux == 0.0:
            intensity = I_location - I_scale * np.log(u)
        else:
            intensity = (
                I_location
                + I_scale * (np.power(u, -1.0 * sigmax_mux) - 1.0) / sigmax_mux
            )

        return intensity


class ConstantRCI:
    GammaRCIModel = GammaRCIModel
    ExponentialRCIModel = ExponentialRCIModel
    WeibullRCIModel = WeibullRCIModel
    ParetoRCIModel = ParetoRCIModel
    GPRCIModel = GPRCIModel
