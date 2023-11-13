import numpy as np
from Library.BLRPRmodel.BLRPRx import BLRPRx as LEGACY_BLRPRx

from pyBL.models import Stat_Props
from pyBL.models import BLRPRx as NEW_BLRPRx

theta = [
    1.0,  # h (scale, unit=hr)
    0.016679,  # lambda
    0.97,  # iota
    1.0,  # sigma_mux
    0.0,  # UNKNOWN
    9.01,  # alpha
    0.901,  # alpha/nu
    0.349,  # kappa
    0.082,  # phi
    1.0,  # c
]
legacy_model = LEGACY_BLRPRx(mode="gamma")
new_model = NEW_BLRPRx()

print(legacy_model.Mean(theta))
print(legacy_model.Cov(theta))
print(legacy_model.Var(theta))
print(np.sqrt(legacy_model.Var(theta)) / legacy_model.Mean(theta))
print()
print(new_model.mean())
print(new_model.covariance())
print(new_model.variance())
print(new_model.get_prop(Stat_Props.CVAR))
