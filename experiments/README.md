# Experiments
This directory contains the implementation of some of the experiments in
the original Adaptive Hierarchical Shrinkage paper.
In order to run these experiments, some extra dependencies must be installed.
This can be done by specifying the `[experiments]` modifier in the installation:

```bash
$ git clone git@github.com:arnegevaert/adaptive-hierarchical-shrinkage.git
$ cd adaptive-hierarchical-shrinkage
$ conda create -n shrinkage python=3.10
$ conda activate shrinkage
$ pip install .[experiments]
```

## Power simulation
This experiment corresponds to section 5.1 in the original paper.
We generate a dataset with 1000 samples and 5 features, one of which is informative.
On this dataset, we train a shrinkage model using each of the 6 shrink modes with a fixed $\lambda$ of 100.0.
Next, we inspect MDI and SHAP values for each shrink mode on the test set.

### Generating results
To generate results in CSV format, run [main.py](power_simulation/main.py).
This script has the following arguments:
- `--n-jobs:` (required) number of jobs to run in parallel. Set to `-1` to use all available cores.
- `--n-replications:` (required) number of replications to run for each setting.
- `--out-dir:` (required) directory to save results to. A subdirectory will be created for each experiment (see `--experiments`).
- `--shrink-modes:` comma-separated list of shrink modes to run. See `--help` for a list of available modes. By default, all modes are run.
- `--experiments:` comma-separated list of experiments to run. See `--help` for a list of available experiments. By default, all experiments are run.

This script will produce a folder with CSV files for each experiment specified in the `--experiments` argument.
Each folder contains 2 CSV files: `importances.csv` and `scores.csv`.
`scores.csv` contains the following columns:
- `relevance:` the relevance of the informative feature (i.e. the signal strength).
- `shrink_mode:` the shrink mode used for this run.
- `ROC AUC:` the ROC AUC score for this run.
- `replication:` the replication number for this run.

`importances.csv` contains the following columns:
- `MDI_0` ... `MDI_4:` the MDI values for each feature.
- `SHAP_0` ... `SHAP_4:` the SHAP values for each feature.
- `relevance:` the relevance of the informative feature (i.e. the signal strength).
- `shrink_mode:` the shrink mode used for this run.
- `replication:` the replication number for this run.

### Plotting results
Plotting the results can be done using the [plot_importances.py](power_simulation/plot_importances.py) and [plot_scores.py](power_simulation/plot_scores.py) scripts.

Both scripts have the following arguments:
- `results_file:` (required) file containing the results. For `plot_importances.py` this is the `importances.csv` file generated by [main.py](power_simulation/main.py). For `plot_scores.py` this is the `scores.csv` file generated by [main.py](power_simulation/main.py).
- `out_dir:` (required) directory to save the plots to.

## General performance
This experiment corresponds to section 5.2 in the original paper.
For each dataset and shrink mode, we select the best value for $\lambda$ using 3-fold CV on the training set.
Next, ROC AUC is reported on the test set for each shrink mode.

For random forests, we additionally generate results for any number of trees ranging from 1 up to 100.
This is done by training a single RF model using 100 trees, and then running inference on the test set using a subset of the trees.
For example, to generate results for 10 trees, we run inference using the first 10 trees of the RF model.
As each tree in the RF model is trained independently from the others, this is equivalent to training a separate RF model using only 10 trees.

### Generating results
To generate results in CSV format, run [main.py](general_performance/main.py).
This script has the following arguments:
- `--n-jobs:` (required) number of jobs to run in parallel. Set to `-1` to use all available cores.
- `--n-replications:` (required) number of replications to run for each setting.
- `--out-dir:` (required) directory to save results to. A subdirectory will be created for each experiment (see `--experiments`).
- `--shrink-modes:` comma-separated list of shrink modes to run. See `--help` for a list of available modes. By default, all modes are run.
- `--experiments:` comma-separated list of experiments to run. See `--help` for a list of available experiments. By default, all experiments are run.

This script will produce a folder with CSV files for each experiment specified in the `--experiments` argument.
Each CSV file corresponds to a dataset.
Each line in a CSV file corresponds to a single run for a single setting of hyperparameters.
The columns are as follows:
- `shrink_mode:` the shrink mode used for this run.
- `lambda:` the value of $\lambda$ used for this run.
- `num_trees:` the number of trees used for this run.
- `ROC AUC:` the ROC AUC score for this run.
- `replication:` the replication number for this run.

### Plotting results
To plot the results, run [plot.py](general_performance/plot.py).
This script has the following arguments:
- `results_dir:` (required) directory containing the results. This should be the same as the `--out-dir` argument used for [main.py](general_performance/main.py).
- `out_dir:` (required) directory to save the plots to.

## SHAP stability
This experiment is not mentioned in the original paper.
Each dataset is split into a train and test set (test set is 1/3 of the data).
Next, 50 random samples of the test set are selected.
For each shrink mode, we select the best value for $\lambda$ using 5-fold CV on the training set.
Next we compute SHAP values for each of the 50 selected samples.
This process is repeated multiple times, allowing for a stability analysis of the resulting SHAP values.

### Generating results
To generate results in CSV format, run [main.py](shap_stability/main.py).
This script has the following arguments:
- `--n-jobs:` (required) number of jobs to run in parallel. Set to `-1` to use all available cores.
- `--n-replications:` (required) number of replications to run for each setting.
- `--out-dir:` (required) directory to save results to. A subdirectory will be created for each dataset (see `--datasets`).
- `--shrink-modes:` comma-separated list of shrink modes to run. See `--help` for a list of available modes. By default, all modes are run.
- `--datasets:` comma-separated list of datasets to use. See `--help` for a list of available datasets. By default, all datasets are used.