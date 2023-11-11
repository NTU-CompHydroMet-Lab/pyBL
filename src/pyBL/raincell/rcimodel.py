from __future__ import annotations

from typing import Optional, Protocol, Union, overload, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore


@runtime_checkable
class IConstantRCI(Protocol):
    rng: np.random.Generator

    def __init__(self, rng: Optional[np.random.Generator] = None):
        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)  # type: ignore

    def set_rng(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

    def get_f1(self, sigmax_mux: float) -> float:
        ...

    def get_f2(self, sigmax_mux: float) -> float:
        ...

    @overload
    def sample_intensity(
        self, mux: float, sigmax_mux: float, n_cells: int
    ) -> npt.NDArray[np.float64]:
        ...

    @overload
    def sample_intensity(
        self,
        mux: float,
        sigmax_mux: float,
    ) -> float:
        ...

    def sample_intensity(
        self, mux: float, sigmax_mux: float, n_cells: Optional[int] = None
    ) -> Union[npt.NDArray[np.float64], float]:
        ...


class GammaRCIModel(IConstantRCI):
    def get_f1(self, sigmax_mux: float) -> float:
        return (sigmax_mux + 1.0) / sigmax_mux

    def get_f2(self, sigmax_mux: float) -> float:
        x2 = sigmax_mux**2
        return (x2 + 3.0 * sigmax_mux + 2.0) / x2

    def sample_intensity(
        self, mux: float, sigmax_mux: float, n_cells: Optional[int] = None
    ) -> Union[npt.NDArray[np.float64], float]:
        intensity = self.rng.gamma(sigmax_mux, mux / sigmax_mux)
        return intensity


class ExponentialRCIModel(IConstantRCI):
    def get_f1(self, sigmax_mux: float) -> float:
        return 2.0

    def get_f2(self, sigmax_mux: float) -> float:
        return 6.0

    @overload
    def sample_intensity(
        self, mux: float, sigmax_mux: float, n_cells: int
    ) -> npt.NDArray[np.float64]:
        ...

    @overload
    def sample_intensity(
        self,
        mux: float,
        sigmax_mux: float,
    ) -> float:
        ...

    def sample_intensity(
        self, mux: float, sigmax_mux: float, n_cells: Optional[int]=None
    ) -> Union[npt.NDArray[np.float64], float]:
        # TODO: Check this c++ code and Legacy pyBL code
        if n_cells is None or n_cells == 1:
            return self._sample_single_intensity(mux, sigmax_mux)
        if n_cells > 1:
            return self._sample_multiple_intensity(mux, sigmax_mux, n_cells)
        else:
            raise TypeError("start and end must be either float or list or np.ndarray")

    def _sample_single_intensity(self, mux: float, sigmax_mux: float) -> float:
        I_shape = 1.0 / sigmax_mux**2
        I_scale = sigmax_mux**2 * mux

        intensity = self.rng.gamma(I_shape, I_scale)

        return intensity

    def _sample_multiple_intensity(
        self, mux: float, sigmax_mux: float, n_cells: int
    ) -> npt.NDArray[np.float64]:
        # TODO: Check this function. Why not fix sigmax_mux to 1.0?
        sigmax_mux = 1.0
        I_shape = 1.0 / sigmax_mux**2
        I_scale = sigmax_mux**2 * mux

        intensity = self.rng.gamma(I_shape, I_scale, size=n_cells)

        return intensity


class WeibullRCIModel(IConstantRCI):
    def get_f1(self, sigmax_mux: float) -> float:
        ex2: float = sp.special.gamma(1.0 + 2.0 / sigmax_mux)
        ex: float = sp.special.gamma(1.0 + 1.0 / sigmax_mux)
        return ex2 / ex**2

    def get_f2(self, sigmax_mux: float) -> float:
        ex3: float = sp.special.gamma(1.0 + 3.0 / sigmax_mux)
        ex: float = sp.special.gamma(1.0 + 1.0 / sigmax_mux)
        return ex3 / ex**3

    def sample_intensity(
        self, start: float, end: float, mux: float, sigmax_mux: float
    ) -> float:
        I_scale = mux / sp.special.gamma(1.0 + 1.0 / sigmax_mux)

        # The weibull distribution in numpy differs from the one in c++
        # See Numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.weibull.html
        # See GSL: https://www.gnu.org/software/gsl/doc/html/randist.html#c.gsl_ran_weibull
        intensity = self.rng.weibull(sigmax_mux) * I_scale

        return intensity


class ParetoRCIModel(IConstantRCI):
    def get_f1(self, sigmax_mux: float) -> float:
        return (sigmax_mux - 1.0) ** 2 / sigmax_mux / (sigmax_mux - 2.0)

    def get_f2(self, sigmax_mux: float) -> float:
        return (sigmax_mux - 1.0) ** 3 / sigmax_mux**2 / (sigmax_mux - 3.0)

    def sample_intensity(
        self, start: float, end: float, mux: float, sigmax_mux: float
    ) -> float:
        I_scale = mux * (sigmax_mux - 1.0) / sigmax_mux

        # The pareto distribution in numpy differs from the one in c++
        # See Numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.pareto.html
        # See GSL: https://www.gnu.org/software/gsl/doc/html/randist.html#c.gsl_ran_pareto
        intensity = self.rng.pareto(sigmax_mux) * I_scale + I_scale

        return intensity


class GPRCIModel(IConstantRCI):
    def get_f1(self, sigmax_mux: float) -> float:
        return 2.0 * (1.0 - sigmax_mux) / (1.0 * 2.0 * sigmax_mux)

    def get_f2(self, sigmax_mux: float) -> float:
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
