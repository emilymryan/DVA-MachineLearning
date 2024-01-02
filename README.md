# Bayesian optimization for li-ion battery aging diagnostics

This script uses Bayesian optimization library that can be downloaded at https://github.com/bayesian-optimization/BayesianOptimization

The provided code is designed to read in voltage and capacity data from pristine anode and cathode, as well as cycled full-cell. Subsequently, the code differentiate voltage with respect to capacity. Following this step, the code employs Bayesian optimization techniques to shift and shrink the pristine differential voltage dataset to achieve the optimal fitting to the cycled full-cell data. 
