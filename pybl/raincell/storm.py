from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Storm:
    def __init__(self, start: float, duration: float, cells: npt.NDArray[np.float64], eta: float, mux: float, gamma: float, beta: float):
        self.start = start
        self.duration = duration
        self.cells = cells
        self.eta = eta
        self.mux = mux
        self.gamma = gamma
        self.beta = beta

