# pyBL

## Description
pyBL is an open-source Python package for stochastic rainfall modelling based upon the randomised Bartlett-Lewis (BL) rectangular pulse model. The BL model is a type of stochastic model that represents rainfall using a Poisson cluster point process.

This package implements the most recent version of the BL model, based upon the state-of-the-art BL model developed in Onof and Wang (2020), and works with standard and widely-used data format. It also includes a number of numerical solvers; this provides the flexibility of developing model fitting strategies. 

In the package, the BL model is separated into three main modules. These are statistical properties calculation, BL model fitting and sampling modules.

The statistical properties calculation module processes the input rainfall data and calculates its standard statistical properties at given timescales. 

The BL model fitting module performs model fitting based upon the re-derived BL equations given in Onof and Wang (2020). A number of numerical solvers, such as Dual Annealing optimization and Nelder-Mead local minimization techniques, are implemented and provided. The combined use of these techniques can ensure efficiency as well as to prevent being drawn to local optima during the solving process.

You may use the sampling module to generate a stochastic rainfall time series at given timescales and for any required data length based upon a fitted BL model.

The design of this package is highly modularised, and the standard CSV data format is used for file exchange between modules. Users can easily incorporrate given modules into their existing applications. 

In addition, a team consisting of researchers from National Taiwan University and Imperial College London will consistently implement the breakthroughs in the BL model to this package so that users would have access to the latest developments.

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

1. Modify the timescales in `./01 Input_data/timeRange.csv`. We need at least 4 different timescales for fitting. For 5 min rainfall time series the combination of 5T,1h,6h,1D is recommended. As for 1 hour rainfall time series 1h,3h,6h,1D should be used. Both combinations are based on experiments thus kindly remind you not to change them whatever you want. If less than 4 timescales are used, please replace the empty with 'NaN'. The same combination with some of the timescales missing is acceptable for model running, however, the missing may affect the performance of model fitting.
    * For the timescale code, please refer to the table below.


    | Second | Minute | Hour | Day | Month | Year |
    | ------ | ------ | ---- | --- | ----- | ---- |
    | S      | T      | H    | D   | M     | Y    |
2. Select the statistical properties for your versions of BL model. 
    * **Note** that the properties must be included in `./Library/Cal_stats_lib/utils.py` or you could directly add your stats prop calculation in `./Library/Cal_stats_lib/utils.py`.
3. run `python3 Runall_main.py`

## In `Runall_main.py`
The whole main script could be divided into three steps which are calculating stats, fitting, and sampling.
You follow the procedure in `Runall_main.py` to understand the workflow of whole BL model and the usage of each sub-module.


## Module
All the modules are in `Library`.

| Module                  | Description |
| ----------------------- | ----------- |
| fitting                 |             |
| utils                   |             |
| sampling.model.BLRPRx   |             |
| intensity.model.expon   |             |
| sampling.merge          |             |
| sampling.sampling_utils |             |


## Calculating Statistical Properties
In the timeRange CSV file, you may specify statistical properties and time scales you would like to use. Available properties are Covariance coefficient, AR-1, and Skewness.

The implementation of calculating statistical properties can be summarized as follows:

0. Read raw data from a CSV file.
1. Read specified statistical properties and time scales from the timeRange CSV file.
2. Resample raw data to 1h data.
3. Calculate mean 1h rain depth and its weight for each calendar month. 
4. Calculate other properties and their weights for each calendar month for time scales specified in the timeRange CSV file.
5. Output the statistical properties and their weights to a CSV file.

### Example code
`Runme01_ Cal_ stats.py`

### Output
1. `./02 Output/staProp_test.csv`
2. `./02 Output/weight_test.csv`

## Fitting a Model
This example takes in a CSV file of statistical properties and corresponding weights calculated from raw data. You may set initial theta in `./config/defalt.yaml`.

The implementation of fitting a model can be summarized as follows:

0. Read statistical properties and corresponding weights from specified CSV files.
1. Set initial theta.
2. Initialize objective function.
3. Initialize fitting model.
4. Find approximate global optimum using Dual Annealing algorithm.
5. Try to find a optimum local minimum using Nelder Mead algorithm.
6. Output the result theta to a CSV file.

### Example code
`Runme02_Fitting.py`
### Output
`./02 Output/Theta.csv`

## Sampling a Model
This example takes in a CSV file of result theta and output sample statistical properties for different time scales.

The implementation of calculating statistical properties can be summarized as follows:

0. Read result theta file from a specified CSV file.
1. Sample storms.
2. Calculate statistical properties of the sampled rainfall time series.
3. Output the statistical properties for each calendar month.

### Example code
`Runme03_Sampling.py`
### Output
`./02 Output/sample_stats.csv`

## Citation
If you use pyBL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

## References
1. G. Marsaglia and W.W. Tsang: The Ziggurat Method for Generating Random Variables. Journal of Statistical Software, vol. 5, no. 8, pp. 1–7, 2000.
2. Doornik, J.A. (2005), “An Improved Ziggurat Method to Generate Normal Random Samples”, mimeo, Nuffield College, University of Oxford.
