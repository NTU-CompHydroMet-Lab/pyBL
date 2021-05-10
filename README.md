# pyBL

## Description
pyBL is an open-source Python package for stochastic rainfall modelling based upon the randomised Bartlett-Lewis (BL) rectangular pulse model. The BL model is a type of stochastic model that represents rainfall using a Poisson cluster point process.

This package implements the most recent version of the BL model, based upon the state-of-the-art BL model developed in Onof and Wang (2020), and works with standard and widely-used data format. It also includes a number of numerical solvers; this provides the flexibility of developing model fitting strategies. 

In the package, the BL model is separated into three main modules. These are statistical properties calculation, BL model calibration, and model sampling (i.e., simulation) modules.

The statistical properties calculation module processes the input rainfall data and calculates its standard statistical properties at given timescales. 

The BL model calibration module conducts the model fitting based upon the re-derived BL equations given in Onof and Wang (2020). A numerical solver, based upon Dual Annealing optimization and Nelder-Mead local minimization techniques, is implemented to ensure efficiency as well as to prevent being drawn to local optima during the solving process.

You may use the sampling module to generate a stochastic rainfall time series at a given timescale and for any required data length based upon a calibrated BL model.

The design of this package is highly modularized, and the standard CSV data format is used for file exchange between modules. It is easily to incorporate given modules into your existing applications. 

In addition, a team consisting of researchers from National Taiwan University and Imperial College London will consistently implement the breakthroughs in the BL model to this package so that you will have access to the latest developments.

## Prerequisites
| Library         | Version   | Website                                    | Reference                               | Description |
| --------------- | --------- | ------------------------------------------ | --------------------------------------- | ----------- |
| Python          | 3.7       | https://www.python.org/                    | Van Rossum and Drake (1995)             |             |
| Numpy           | 1.20.1    | https://numpy.org/                         | Van Der Walt et al.(2011)               |             |
| SciPy           | 1.6.1     | https://www.scipy.org/                     | Jones et al. (2001)                     |             |
| Pandas          | 1.2.3     | https://pandas.pydata.org/                 | McKinney  (2010)                        |             |
| statsmodels     | 0.12.2    | https://www.statsmodels.org/stable/#       | Seabold and Skipper and Perktold (2010) |             |
| certifi         | 2020.12.5 | https://github.com/certifi/python-certifi  | Reitz (2011)                            |             |
| python-dateutil | 2.8.1     | https://dateutil.readthedocs.io/en/stable/ | Niemeyer (2003)                         |             |
| patsy           | 0.5.1     | https://patsy.readthedocs.io/en/latest/    | Smith (2012)                            |             |
| pytz            | 2021.1    | https://github.com/stub42/pytz             | Bishop (2004)                           |             |
| six             | 1.15.0    | https://six.readthedocs.io/                | Peterson (2010)                         |             |
| zignor          | 0.1.8     | https://github.com/jameslao/zignor-python  | Lao (2015)                              |             |
| matplolib       | 3.0.3     | https://matplotlib.org/                    | Hunter (2007)                           |             |


## Installation
Clone this repo or enter the following texts in your command line.
```git clone https://github.com/NTU-CompHydroMet-Lab/pyBL.git```

## Getting Started

1. modified the time scale in `timerange.csv`
    * For the time scale code, please refer to the table


    | Second | Minute | Hour | Day | Month | Year |
    | ------ | ------ | ---- | --- | ----- | ---- |
    | S      | T      | H    | D   | M     | Y    |
2. Select the statistical properties for your versions of BL model. 
    * **Note** that the properties must be included in `utils.py` or you could directly add your stats prop calculation in `utils.py`.
3. run `python3 main.py`

## In `main.py`
The whole main script could be divided into three steps which are calculating stats, fitting, and sampling.
You follow the procedure in `main.py` to understand the workflow of whole BL model and the usage of each sub-module.


## Module
| Module                  | Description |
| ----------------------- | ----------- |
| fitting                 |             |
| utils                   |             |
| sampling.model.BLRPRx   |             |
| intensity.model.expon   |             |
| sampling.merge          |             |
| sampling.sampling_utils |             |

## Documentation
<!--參考numpy, pandas的documentation-->
### fitting
### fitting.fitting.Annealing

> <span style="color:deeppink">**Annealing**</span>(**theta** = List, **obj_func** = Func, **month** = int, **timeScaleList** = List, **staFile_path** = Str, **weightFile_path** = Str)

Optimize the theta, the parameter of BL model by dual annealing which is a global optimization method provided by scipy
- Parameters:
    - theta: list
        A list of parameters
    - obj_func: function method
        The model equation
    - month: int
        The index to assign the stats properties from stats file and weight file
    - timeScaleList: list
    a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    - staFile_path: string
    The file path of the statistical properties
    - weightFile_path: string
    The file path of the weight(uncertainty) of the statistical properties
- Returns:
    -  (theta, ret.fun): (list, float)
    The dict contains the optimized theta and the score for checking if the optimizatiom is correct
    
### fitting.fitting.Basinhopping

> <span style="color:deeppink">**Basinhopping**</span>(**theta** = List, **obj_func** = Func, **month** = int, **timeScaleList** = List, **staFile_path** = Str, **weightFile_path** = Str)

Optimize the theta, the parameter of BL model by basinhopping which is a global optimization method provided by scipy
- Parameters:
    - theta: list
        A list of parameters
    - obj_func: function method
        The model equation
    - month: int
        The index to assign the stats properties from stats file and weight file
    - timeScaleList: list
    a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    - staFile_path: string
    The file path of the statistical properties
    - weightFile_path: string
    The file path of the weight(uncertainty) of the statistical properties
- Returns:
    -  (theta, ret.fun): (list, float)
    The dict contains the optimized theta and the score for checking if the optimizatiom is correct

### fitting.fitting.Nelder_Mead

> <span style="color:deeppink">**Nelder_Mead**</span>(**theta** = List, **obj_func** = Func, **month** = int, **timeScaleList** = List, **staFile_path** = Str, **weightFile_path** = Str)

Optimize the theta, the parameter of BL model by local minimization optimization methods provided by scipy
- Parameters:
    - theta: list
        A list of parameters
    - obj_func: function method
        The model equation
    - month: int
        The index to assign the stats properties from stats file and weight file
    - timeScaleList: list
    a list contains different ratio based on "1 Hour = 1" to fulfill the parameters
    - staFile_path: string
    The file path of the statistical properties
    - weightFile_path: string
    The file path of the weight(uncertainty) of the statistical properties
- Returns:
    -  (theta, ret.fun): (list, float)
    The dict contains the optimized theta and the score for checking if the optimizatiom is correct


### sampling
### sampling.model.BLRPRx.SampleStorm
> <span style="color:deeppink">**SampleStorm**</span>(**theta**=None, **simulation_period**=10000, **start_time**=datetime.now())
- Parameters:
    - theta: list
        The list should be composed of six float variables in correct order, which is $[\lambda, \iota, \alpha, \dfrac{\alpha}{\nu}, \kappa, \phi]$.
        $\lambda:$ storm arrival rate
        $\iota:$ ratio of mean cell intensity to eta
        $\alpha:$ shape parameter for gamma distribution
        $\nu:$ scale parameter for gamma distribution
        $\kappa:$ cell arrival rate
        $\phi:$ storm termination rate
    - simulation_period: int
        total time length with hour as unit of simulation. default value of simulation_period is 10000
    - start_time: *datetime* object
        start time of simulation, default value of start_time is datetime.now()
```
from datetime import datetime

theta = [0.0151, 0.2385 1.0567, 4.0581, 0.6983, 0.0287]
storms = SampleStorm(theta, 10000, datetime.now())
```
        
- Returns:
    - out: *Storm* object
### sampling.merge.MergeCells
> <span style="color:deeppink">**MergeCells**</span>(**storms**=None, **time_scale**='5T')
- Parameters:
    - storms: list
        a list that contains *Storm* object
    - time_scale: str
        the unit of time of rainfall time series, the available parameters are ['5T', '1H', '6H', '1D'], default value of time_scale is '5T'.
- Returns:
    - out: *pandas.Series* object
        a rainfall intensity time series
```
from sampling.model.BLRPRx import SampleStorm
from sampling.merge import MergeCells
from datetime import datetime

theta = [0.0151, 0.2385 1.0567, 4.0581, 0.6983, 0.0287]
storms = SampleStorm(theta, 10000, datetime.strptime('2020-01-01', '%Y-%m-%d'
ts = MergeCells(storms, '5T')
```
### sampling.merge.ConcatenateCells
> <span style="color:deeppink">**ConcatenateCells**</span>(**storms**=None, **time_scale**='5T')
- Parameters:
    - storms: list
        a list that contains *Storm* object generated individually from 12 months' data
    - time_scale: str
        the unit of time of rainfall time series, the available parameters are ['5T', '1H', '6H', '1D'], default value of time_scale is '5T'.
- Returns:
    - out: *pandas.DataFrame* object
        contains several years of rainfall
```
from sampling.model.BLRPRx import SampleStorm
from sampling.merge import ConcatenateCells
from datetime import datetime

thetas = [[0.0151302303, 0.2385468898, 1.0567517930, 4.0581327313, 0.6983167827, 0.0287377630],
          [0.0128651067, 0.2144378007, 0.9311980491, 5.6977092457, 0.6210653123, 0.0216480507],
          [0.0167114492, 0.2361159050, 1.2354096661, 5.0782508215, 0.6323218399, 0.0305174937],
          [0.0185657505, 0.2731468849, 1.1142537224, 5.3049163169, 0.3717318391, 0.0262894281],
          [0.0131596859, 0.7763544893, 0.5385350594, 4.8540684521, 0.2118119486, 0.0283355427],
          [0.0195250688, 5.4079188215, 0.3905713502, 1.0524090891, 0.0101463190, 0.3585225909],
          [0.0202012609, 0.9256232588, 0.5039770144, 6.8805273843, 0.4070711276, 0.0858894054],
          [0.0135665969, 0.9527049530, 0.5986612357, 8.5229950222, 0.1437084709, 0.0216822244],
          [0.0144151456, 0.8642425983, 0.5841580273, 5.4492217783, 0.1936114469, 0.0298269345],
          [0.0119492179, 0.3302936737, 1.0235468679, 5.8169710262, 0.4943459566, 0.0236522015],
          [0.0148426929, 0.2349993504, 0.8098682540, 6.2656746152, 0.6567194811, 0.0239520055],
          [0.0135993788, 0.2267651636, 1.0440038194, 4.2939621019, 1.0253823787, 0.0317435744]]

multi_storms = []
for subt in thetas:
    storms = SampleStorm(theta=subt, simulation_period=simulation_period, start_time=start_time)
    multi_storms.append(storms)
    
df = ConcatenateCells(multi_storms, time_scale='1H')
```

### sampling.sampling_utils.ExportStorms
> <span style="color:deeppink">**ExportStorms**</span>(**storms**=None, **file_name**=None)
- Parameters:
    - storms: list
        a list of *Storm* object
    - file_name: str
        the file_name should end in '.pkl'
- Returns:
    - out: int
### sampling.sampling_utils.gamma
> <span style="color:deeppink">**gamma**</span>(**shape**=None, **scale**=None)
- Parameters:
    - shape: float
    - scale: float
- Returns:
    - out: float
```
from sampling.sampling_utils import gamma
x = gamma(1.0, 1.0)
```
### sampling.sampling_utils.OptBins
> <span style="color:deeppink">**OptBins**</span>(**target**=None, **maxBins**=10)
- Parameters:
    - target: list or numpy.array
    - maxBins: int
        default value of maxBins is 10
- Returns:
    - out: int

## Citation
If you use pyBL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

## Contact

## References
1. G. Marsaglia and W.W. Tsang: The Ziggurat Method for Generating Random Variables. Journal of Statistical Software, vol. 5, no. 8, pp. 1–7, 2000.
2. Doornik, J.A. (2005), “An Improved Ziggurat Method to Generate Normal Random Samples”, mimeo, Nuffield College, University of Oxford.