import argparse
import yaml
from pathlib import Path
from _run_experiment import run_experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

EXPERIMENTS = ["strobl_rf", "strobl_dt"]

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

    output_dir = Path(config["output_dir"]) / "power_simulation"
    shrink_modes = config["shrink_modes"]
    lmb = config["power_simulation"]["lambda"]
    relevances = config["power_simulation"]["relevances"]
    n_replications = config["power_simulation"]["n_replications"]

    print(f"Running experiments: {args.experiments}")
    print(f"Number of replications: {n_replications}")
    print(f"Shrink modes: {shrink_modes}")
    print(f"Lambda: {lmb}")
    print(f"Relevances: {relevances}")

    if "strobl_dt" in args.experiments:
        print("Running Strobl Decision Tree experiment...")
        run_experiment(
            lmb,
            relevances,
            shrink_modes,
            DecisionTreeClassifier(),
            args.n_jobs,
            n_replications,
            output_dir,
            "decision_tree",
        )
    if "strobl_rf" in args.experiments:
        print("Running Strobl Random Forest experiment...")
        run_experiment(
            lmb,
            relevances,
            shrink_modes,
            RandomForestClassifier(),
            args.n_jobs,
            n_replications,
            output_dir,
            "random_forest",
        )
