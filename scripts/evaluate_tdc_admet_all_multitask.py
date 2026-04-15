"""Evaluate predictions from multitask Chemprop models on all TDC ADMET datasets."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, r2_score

from tdc_constants import ADMET_ALL_SMILES_COLUMN


def evaluate_tdc_admet_all_multitask(data_dir: Path, preds_dir: Path, save_dir: Path) -> None:
    """Evaluate predictions from multitask Chemprop models on all TDC ADMET datasets.

    :param data_dir: A directory containing the multitask TDC ADMET datasets.
    :param preds_dir: A directory containing the predictions from Chemprop models trained on the multitask TDC ADMET datasets.
    :param save_dir: A directory where the evaluation results will be saved.
    """
    # Load data
    regression_data = pd.read_csv(data_dir / "admet_regression.csv").set_index(ADMET_ALL_SMILES_COLUMN)
    classification_data = pd.read_csv(data_dir / "admet_classification.csv").set_index(ADMET_ALL_SMILES_COLUMN)

    # Load predictions for each dataset
    regression_preds_list = [
        pd.read_csv(path).set_index(ADMET_ALL_SMILES_COLUMN)
        for path in preds_dir.glob("admet_regression/**/test_predictions.csv")
    ]
    classification_preds_list = [
        pd.read_csv(path).set_index(ADMET_ALL_SMILES_COLUMN)
        for path in preds_dir.glob("admet_classification/**/test_predictions.csv")
    ]

    # Set up save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate predictions for each dataset
    task_to_results = defaultdict(lambda: defaultdict(list))

    for classification_preds in classification_preds_list:
        matched_classification_data = classification_data.loc[classification_preds.index]

        for task in classification_preds.columns:
            valid_mask = matched_classification_data[task].notna()
            valid_data = matched_classification_data[valid_mask]
            valid_preds = classification_preds[valid_mask]

            task_to_results[task]["roc_auc"].append(roc_auc_score(valid_data[task], valid_preds[task]))
            task_to_results[task]["prc_auc"].append(average_precision_score(valid_data[task], valid_preds[task]))

    task_to_summary_results = {
        task: {
            "ROC AUC Mean": np.mean(task_to_results[task]["roc_auc"]),
            "ROC AUC Std": np.std(task_to_results[task]["roc_auc"]),
            "PRC AUC Mean": np.mean(task_to_results[task]["prc_auc"]),
            "PRC AUC Std": np.std(task_to_results[task]["prc_auc"]),
        }
        for task in classification_data.columns
    }

    classification_results = pd.DataFrame(task_to_summary_results).T.reset_index(names="Task")

    pd.DataFrame(classification_results).to_csv(save_dir / "admet_classification_results.csv", index=False)

    for regression_preds in regression_preds_list:
        matched_regression_data = regression_data.loc[regression_preds.index]

        for task in regression_preds.columns:
            valid_mask = matched_regression_data[task].notna()
            valid_data = matched_regression_data[valid_mask]
            valid_preds = regression_preds[valid_mask]

            task_to_results[task]["mae"].append(mean_absolute_error(valid_data[task], valid_preds[task]))
            task_to_results[task]["r2"].append(r2_score(valid_data[task], valid_preds[task]))

    task_to_summary_results = {
        task: {
            "MAE Mean": np.mean(task_to_results[task]["mae"]),
            "MAE Std": np.std(task_to_results[task]["mae"]),
            "R2 Mean": np.mean(task_to_results[task]["r2"]),
            "R2 Std": np.std(task_to_results[task]["r2"]),
        }
        for task in regression_data.columns
    }

    regression_results = pd.DataFrame(task_to_summary_results).T.reset_index(names="Task")

    pd.DataFrame(regression_results).to_csv(save_dir / "admet_regression_results.csv", index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(evaluate_tdc_admet_all_multitask)
