import pandas as pd
import yaml
from imodels.util.data_util import get_clean_dataset

REPLACE = {
    "diabetes-clf": {
        "tested_positive": 1,
        "tested_negative": 0,
    },
    "german": {
        "good": 1,
        "bad": 0,
    }
}


if __name__ == "__main__":
    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["data_dir"]
    for dataset_type in ["classification", "regression"]:
        for dataset_name, dataset_dict in config["datasets"][dataset_type].items():
            id = dataset_dict.get("id")
            source = dataset_dict.get("source")
            X, y, _ = get_clean_dataset(id, source)
            if dataset_name in REPLACE:
                y = pd.Series(y).cat.rename_categories(REPLACE[dataset_name]).values
            if dataset_name in ["red-wine", "abalone"]:
                y = y.astype(int)

            pd.DataFrame(X).to_csv(
                f"{data_dir}/{dataset_name}_X.csv", index=False, header=False
            )
            pd.DataFrame(y).to_csv(
                f"{data_dir}/{dataset_name}_y.csv", index=False, header=False
            )
