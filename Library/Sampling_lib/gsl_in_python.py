import math
import zignor
from scipy.stats import expon, randint, uniform

def uniform_pos():
    # This function returns a positive double precision floating point number
    # uniformly distributed in the range (0,1).
    x = uniform.rvs()
    if x > 0 and x < 1:
        return x
    else:
        return uniform_pos()

def gamma(a, b):
    # assume a > 0
    if a < 1:
        u = uniform_pos()
        while u < 0.1 or u > 0.99999:
            u = uniform_pos()
        return gamma(1 + a, b) * (u ** (1 / a))

    d = a - 1 / 3
    c = 1 / 3 / math.sqrt(d)

    while True:
        x = zignor.randn(1)[0]
        v = 1 + c * x
        while v <= 0:  # and x < -math.sqrt(9*a-3)
            x = zignor.randn(1)[0]
            v = 1 + c * x

        v = v * v * v
        u = uniform_pos()

        if u < (1 - 0.0331 * x * x * x * x):
            # return d * v
            break

        if math.log(u) < (0.5 * x * x + d * (1 - v + math.log(v))):
            # return d * v
            break

    return b * d * v
