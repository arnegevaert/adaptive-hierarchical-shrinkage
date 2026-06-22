import argparse
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import yaml
from _run_single_replication import run_single_replication
from sklearn.model_selection import train_test_split
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int)
    args = parser.parse_args()

    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_replications = config["n_replications"]
    output_dir = Path(config["output_dir"]) / "shap_stability"
    data_dir = Path(config["data_dir"])
    shrink_modes = config["shrink_modes"]
    clf_datasets = config["datasets"]["classification"]
    lambdas = config["lambdas"]

    os.makedirs(output_dir, exist_ok=True)

    for ds_name in clf_datasets.keys():
        X = np.loadtxt(os.path.join(data_dir, f"{ds_name}_X.csv"), delimiter=",")
        y = np.loadtxt(os.path.join(data_dir, f"{ds_name}_y.csv"), delimiter=",")
        feature_names = [
            f"feature_{i}" for i in range(X.shape[1])
        ]  # TODO: load feature names from file if available

        y = y.astype(int)

        # Separate 50 samples for computing Shapley values
        X, X_shap, y, y_shap = train_test_split(
            X, y, test_size=50, random_state=0, stratify=y
        )

        results: List[pd.DataFrame] = []
        if args.n_jobs == 1:
            prog = trange(n_replications, desc=f"Running {ds_name}")
            for _ in prog:
                results.append(
                    run_single_replication(
                        feature_names,
                        shrink_modes,
                        lambdas,
                        X,
                        y,
                        X_shap,
                        y_shap,
                    )
                )
        else:
            results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
                joblib.delayed(run_single_replication)(
                    feature_names,
                    shrink_modes,
                    lambdas,
                    X,
                    y,
                    X_shap,
                    y_shap,
                )
                for _ in range(n_replications)
            )  # type: ignore

        # Add replication column to each dataframe
        for i, df in enumerate(results):
            df["replication"] = i

        all_results = pd.concat(results)
        all_results.to_csv(os.path.join(output_dir, f"{ds_name}.csv"), index=False)

