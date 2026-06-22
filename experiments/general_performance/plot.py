import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

if __name__ == "__main__":
    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["output_dir"]) / "general_performance"
    output_dir = Path(config["output_dir"]) / "general_performance"
    os.makedirs(output_dir, exist_ok=True)

    for subdir in ["classification_dt", "classification_rf", "regression"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        if os.path.isdir(os.path.join(results_dir, subdir)):
            dfs = []
            for filename in os.listdir(os.path.join(results_dir, subdir)):
                print(f"Processing {subdir}/{filename}...")
                if filename.endswith(".csv"):
                    ds_name = filename.split(".")[0]
                    df = pd.read_csv(os.path.join(results_dir, subdir, filename))
                    df["dataset"] = ds_name
                    dfs.append(df)
            total_df = pd.concat(dfs, ignore_index=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=total_df,
                x="dataset",
                y="ROC AUC" if "ROC AUC" in total_df.columns else "R2",
                hue="shrink_mode",
                ax=ax,
                notch=True,
                showcaps=False,
                flierprops={"marker": "x"},
            )

            fig.savefig(
                os.path.join(os.path.join(output_dir), f"{subdir}.svg"),
                bbox_inches="tight",
            )
