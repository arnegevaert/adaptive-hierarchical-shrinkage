import argparse
import os
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import yaml
from pathlib import Path

from _run_experiment import run_experiment

EXPERIMENTS = ["classification_dt", "classification_rf", "regression"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
    )
    args = parser.parse_args()

    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_replications = config["n_replications"]
    output_dir = Path(config["output_dir"]) / "general_performance"
    data_dir = Path(config["data_dir"])
    shrink_modes = config["shrink_modes"]
    clf_datasets = config["datasets"]["classification"]
    reg_datasets = config["datasets"]["regression"]
    lambdas = config["lambdas"]

    print(f"Running experiments: {args.experiments}")
    print(f"Number of replications: {n_replications}")
    print(f"Shrink modes: {shrink_modes}")
    print(f"Classification datasets: {list(clf_datasets.keys())}")
    print(f"Regression datasets: {list(reg_datasets.keys())}")
    print(f"Lambdas: {lambdas}")

    np.seterr(all="raise")

    os.makedirs(output_dir, exist_ok=True)

    if "classification_dt" in args.experiments:
        run_experiment(
            clf_datasets,
            DecisionTreeClassifier(),
            shrink_modes,
            lambdas,
            "classification",
            roc_auc_score,
            args.n_jobs,
            n_replications,
            output_dir,
            data_dir,
            "classification_dt",
        )
    if "classification_rf" in args.experiments:
        run_experiment(
            clf_datasets,
            RandomForestClassifier(),
            shrink_modes,
            lambdas,
            "classification",
            roc_auc_score,
            args.n_jobs,
            n_replications,
            output_dir,
            data_dir,
            "classification_rf",
        )
    if "regression" in args.experiments:
        run_experiment(
            reg_datasets,
            DecisionTreeRegressor(),
            shrink_modes,
            lambdas,
            "regression",
            r2_score,
            args.n_jobs,
            n_replications,
            output_dir,
            data_dir,
            "regression",
        )
