import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path


if __name__ == "__main__":
    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["output_dir"]) / "general_performance"
    output_dir = Path(config["output_dir"]) / "general_performance" / "plot"

    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            ds_name = filename.split(".")[0]
            df = pd.read_csv(os.path.join(results_dir, filename))
            score_key = df.columns[-1]
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="lambda", y=score_key, hue="shrink_mode")
            plt.xscale("log")
            plt.savefig(
                os.path.join(out_dir, f"{ds_name}.svg"),
                bbox_inches="tight",
            )
