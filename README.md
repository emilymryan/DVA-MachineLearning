# Bayesian optimization and gradient descent for li-ion battery aging diagnostics

This script uses Bayesian optimization library that can be downloaded at https://github.com/bayesian-optimization/BayesianOptimization

The provided python scripts are designed to read in voltage and capacity data from pristine anode and cathode, as well as cycled full-cell. Subsequently, the scipts differentiate voltage with respect to capacity. Following this step, the two scripts employ Bayesian optimization (BO) and gradient descent (GD) techniques respectively to shift and shrink the pristine differential voltage dataset to achieve the optimal fitting to the cycled full-cell data. 

The raw data that these scripts uses are provided by the Argonne National Lab, and not included in the repository. Please send requests or questions to jack7z@bu.edu
