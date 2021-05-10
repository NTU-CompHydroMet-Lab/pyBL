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

## Citation
If you use pyBL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

## Contact

## References
1. G. Marsaglia and W.W. Tsang: The Ziggurat Method for Generating Random Variables. Journal of Statistical Software, vol. 5, no. 8, pp. 1–7, 2000.
2. Doornik, J.A. (2005), “An Improved Ziggurat Method to Generate Normal Random Samples”, mimeo, Nuffield College, University of Oxford.