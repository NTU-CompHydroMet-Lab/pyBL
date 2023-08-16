import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp


def OPTBINS(target, maxBins):
    """
    :param target: array with size (1, N)
    :param maxBins: int
    :return optBins: int
    """
    if len(target.shape) > 1:
        print("The dimensions of input data must be (1, N)!")
        return

    N = target.shape[0]
    logp = np.zeros(
        [
            maxBins,
        ]
    )

    for b in range(1, maxBins):
        n = np.histogram(target, b)[0]
        part1 = N * math.log(b) + sp.loggamma(b / 2) - sp.loggamma(N + b / 2)
        part2 = -b * sp.loggamma(1 / 2) + np.sum(
            sp.loggamma(n + np.array([0.5] * len(n)))
        )
        logp[b] = part1 + part2

    # maxScore = np.max(logp)
    optBins = np.where(logp == np.max(logp))

    return optBins[0][0]
