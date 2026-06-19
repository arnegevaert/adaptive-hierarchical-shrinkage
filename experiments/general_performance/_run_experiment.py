import os
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from _run_single_replication import run_single_replication
from _util import TreeBasedModel
from imodels.util.data_util import get_clean_dataset
from joblib import Parallel, delayed
from tqdm import tqdm


def run_experiment(
    datasets: Dict[str, Dict[str, str]],
    base_estimator: TreeBasedModel,
    shrink_modes: List[str],
    lambdas: List[float],
    problem_type: str,
    score_fn: Callable,
    n_jobs: int,
    n_replications: int,
    out_dir: Path,
    data_dir: Path,
    exp_name: str,
):
    prog = tqdm(datasets.keys())
    prog.set_description(f"Running {exp_name}")

    out_path = os.path.join(out_dir, exp_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for ds_name in prog:
        prog.set_postfix({"dataset": ds_name})
        X = np.loadtxt(os.path.join(data_dir, f"{ds_name}_X.csv"), delimiter=",")
        y = np.loadtxt(os.path.join(data_dir, f"{ds_name}_y.csv"), delimiter=",")

        # Each call to run_single_replication will return a list of records
        # Each record corresponds to a line in a CSV file
        results: List[List[Dict[str, float | int | str]]] = []
        if n_jobs == 1:
            for _ in range(n_replications):
                results.append(
                    run_single_replication(
                        X,
                        y,
                        base_estimator,
                        shrink_modes,
                        lambdas,
                        problem_type,
                        score_fn,
                    )
                )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(run_single_replication)(
                    X,
                    y,
                    base_estimator,
                    shrink_modes,
                    lambdas,
                    problem_type,
                    score_fn,
                )
                for _ in range(n_replications)
            )  # type: ignore

        # For each replication, add a column indicating the replication number
        # and save the results to a CSV file
        all_records = []
        for i, records in enumerate(results):
            for record in records:
                record["replication"] = i
                all_records.append(record)
        pd.DataFrame(all_records).to_csv(
            os.path.join(out_path, f"{ds_name}.csv"), index=False
        )
