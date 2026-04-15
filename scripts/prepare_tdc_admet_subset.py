"""Download and prepare selected Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets."""

from pathlib import Path

import pandas as pd
from tdc.benchmark_group import admet_group
from tqdm import tqdm

from tdc_constants import ADMET_GROUP_SEEDS


def prepare_tdc_admet_subset(raw_data_dir: Path, save_dir: Path, datasets: list[str]) -> None:
    """Download and prepare selected TDC ADMET Benchmark Group datasets.

    :param raw_data_dir: Directory where raw TDC data will be stored.
    :param save_dir: Directory where formatted CSVs will be saved.
    :param datasets: Dataset names from ADMET_Group (e.g., caco2_wang, hia_hou).
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=raw_data_dir)

    for data_name in tqdm(datasets):
        benchmark = group.get(data_name)

        train_val_data = benchmark["train_val"].copy()
        test_data = benchmark["test"].copy()
        train_val_data["split"] = "train"
        test_data["split"] = "test"

        for seed in ADMET_GROUP_SEEDS:
            dataset_dir = save_dir / data_name / str(seed)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            train, valid = group.get_train_valid_split(benchmark=benchmark["name"], split_type="default", seed=seed)
            train_val_data.loc[train.index, "split"] = "train"
            train_val_data.loc[valid.index, "split"] = "val"

            combined_data = pd.concat([train_val_data, test_data]).reset_index(drop=True)
            combined_data["dataset"] = data_name
            combined_data["seed"] = seed
            combined_data.to_csv(dataset_dir / "data.csv", index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(prepare_tdc_admet_subset)
