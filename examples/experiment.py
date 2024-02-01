from pybl.timeseries import IndexedShapshot
from pybl.timeseries.indexsnapshot import _ishapshot_check as check
import numpy as np

ist = IndexedShapshot([0.5, 1.5, 2.5, 3.5], [1, 2, 3, 4])
print(f"Result time: {ist.time}")
print(f"Result intensity: {ist.intensity}")

new_ist = ist[0:4.5]
print(new_ist.time)
print(new_ist.intensity)

