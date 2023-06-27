import pandas as pd
import numpy as np
import pdb
from utils import *

df = pd.read_csv('./result/Jan_5T_1.csv', index_col = 0)
df.index = pd.to_datetime(df.index)
oneH = change_timescale(df, '1H')
sixH = change_timescale(df, '6H')
oneD = change_timescale(df, '1D')
pdb.set_trace()