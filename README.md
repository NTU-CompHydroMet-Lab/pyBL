# pyBL


An open souce Python package for stochastic rainfall modelling based upon a Randomised Bartlett-Lewis Rectangular Pulse Model

## Installation

Clone this repo or enter the following texts in your command line.
`git clone https://github.com/NTU-CompHydroMet-Lab/pyBL.git`

## Getting Started

1. modified the time scale in `timerange.csv`
    * For the time scale code, please refer to the table


    | Second | Minute | Hour | Day | Month | Year |
    | -------- | -------- | -------- |-------- | -------- | -------- |
    | S| T | H| D | M | Y |
2. Select the statistical properties for your versions of BL model. 
    * **Notice** that the propties must be included in `utils.py` or you could directly add your stats prop calculation in `utils.py`.
3. run `python3 main.py`

## Citing pyBL
If you use pyBL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.
