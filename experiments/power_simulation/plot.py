import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

if __name__ == "__main__":
    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["output_dir"]) / "power_simulation"
    output_dir = Path(config["output_dir"]) / "power_simulation"
    os.makedirs(output_dir, exist_ok=True)

    for subdir in ["decision_tree", "random_forest"]:
        print(f"Processing {subdir}...")
        files = os.listdir(os.path.join(results_dir, subdir))
        if not "scores.csv" in files or not "importances.csv" in files:
            print(
                f"Missing scores.csv or importances.csv in {results_dir}/{subdir}. Skipping..."
            )
            continue

        scores_df = pd.read_csv(os.path.join(results_dir, subdir, "scores.csv"))
        importances_df = pd.read_csv(
            os.path.join(results_dir, subdir, "importances.csv")
        )

        relevances = scores_df["relevance"].unique()
        for relevance in relevances:
            # Melt the importances dataframe to long format
            importances_melted = importances_df[importances_df["relevance"] == relevance].melt(
                id_vars=["relevance", "shrink_mode", "replication"],
                value_name="importance",
            )
            # We want to have a column for the importance type (MDI, SHAP) and a column for the feature name
            # The column names in the scores dataframe are of the form "MDI_feature1", "SHAP_feature2", etc.
            # We can split the column names on the first underscore to get the importance type and feature name
            importances_melted[["importance_type", "feature"]] = importances_melted[
                "variable"
            ].str.split("_", n=1, expand=True)

            for importance_type in ["MDI", "SHAP"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(
                    data=importances_melted[importances_melted["importance_type"] == importance_type],
                    x="feature",
                    y="importance",
                    hue="shrink_mode",
                    ax=ax,
                    notch=False,
                    showcaps=False,
                    flierprops={"marker": "x"},
                )
                print(f"Saving {importance_type} relevance plot for {subdir} with relevance {relevance}...")
                for ext in ["svg", "png"]:
                    fig.savefig(
                        os.path.join(
                            os.path.join(output_dir, subdir),
                            f"{importance_type}_relevance_{relevance}.{ext}",
                        ),
                        bbox_inches="tight",
                )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=scores_df,
            x="relevance",
            y="ROC AUC",
            hue="shrink_mode",
            ax=ax,
            notch=False,
            showcaps=False,
            flierprops={"marker": "x"},
        )
        print(f"Saving ROC AUC plot for {subdir}...")
        for ext in ["svg", "png"]:
            fig.savefig(
                os.path.join(
                    os.path.join(output_dir, subdir),
                    f"ROC_AUC.{ext}",
                ),
                bbox_inches="tight",
            )
    print("Done.")
