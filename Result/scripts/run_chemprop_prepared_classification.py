"""Run Chemprop on mentor-provided prepared classification datasets and build result tables."""

from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from chemprop.utils.utils import make_mol
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score


def _resolve_prepared_joblib(prepared_dir: Path, dataset: str, prepared_extra_dir: Path | None = None) -> Path:
    candidates = [
        prepared_dir / f"{dataset}.joblib",
        prepared_dir / "prepared" / f"{dataset}.joblib",
    ]
    if prepared_extra_dir is not None:
        candidates.extend(
            [
                prepared_extra_dir / f"{dataset}.joblib",
                prepared_extra_dir / "prepared" / f"{dataset}.joblib",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {dataset}.joblib in searched dirs: "
        + ", ".join(sorted({str(p.parent) for p in candidates}))
    )


def _select_smiles_target_columns(df: pd.DataFrame, target_col: str = "") -> tuple[str, str]:
    smiles_candidates = ["smiles", "SMILES", "Drug"]
    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError(f"Could not find smiles column in dataframe columns: {list(df.columns)}")

    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"target_col={target_col} not in dataframe columns")
        return smiles_col, target_col

    label_candidates = ["Y", "target", "label", "labels"]
    for c in label_candidates:
        if c in df.columns:
            return smiles_col, c

    blocked = {"smiles", "SMILES", "Drug", "graph"}
    id_like = {c for c in df.columns if ("id" in c.lower() or "split" in c.lower())}
    candidates = [c for c in df.columns if c not in blocked and c not in id_like]
    if len(candidates) != 1:
        raise ValueError(
            "Could not auto-select a single target column. "
            f"Set --target_col explicitly. Candidate target columns: {candidates}"
        )
    return smiles_col, candidates[0]


def _load_prepared_dataset(
    dataset: str,
    prepared_dir: Path,
    prepared_src_dir: Path,
    prepared_extra_dir: Path | None = None,
) -> tuple[pd.DataFrame, str, list[dict[str, list[int]]]]:
    prepared_src_dir = prepared_src_dir.resolve()
    if str(prepared_src_dir) not in sys.path:
        sys.path.insert(0, str(prepared_src_dir))

    dataset_path = _resolve_prepared_joblib(
        prepared_dir=prepared_dir,
        dataset=dataset,
        prepared_extra_dir=prepared_extra_dir,
    )
    obj = joblib.load(dataset_path)

    if not hasattr(obj, "data") or not hasattr(obj, "splits") or not hasattr(obj, "task"):
        raise ValueError(f"{dataset_path} is not a recognized prepared Dataset object.")

    if str(obj.task) != "classification":
        raise ValueError(f"{dataset}: expected classification task, got {obj.task}")

    splits = obj.splits
    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError(f"{dataset}: expected list of split dicts, got {type(splits)}")

    return obj.data.copy(), str(obj.task), splits


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def _pick_prediction_column(preds_df: pd.DataFrame, smiles_col: str) -> str:
    candidates = [c for c in preds_df.columns if c != smiles_col]
    if not candidates:
        raise ValueError("Could not find prediction column in Chemprop predictions output.")

    # Use the first numeric non-smiles column.
    for c in candidates:
        if pd.api.types.is_numeric_dtype(preds_df[c]):
            return c
    return candidates[0]


def _filter_invalid_smiles_and_fix_splits(
    df: pd.DataFrame,
    splits: list[dict[str, list[int]]],
    smiles_col: str,
) -> tuple[pd.DataFrame, list[dict[str, list[int]]], int]:
    def _is_chemprop_valid(s: str) -> bool:
        # Match Chemprop's own molecule parsing rules to avoid training-time crashes.
        try:
            _ = make_mol(s, keep_h=False, add_h=False, ignore_stereo=False, reorder_atoms=False)
            return True
        except Exception:
            return False

    valid_mask = df[smiles_col].astype(str).apply(_is_chemprop_valid).to_numpy()
    num_removed = int((~valid_mask).sum())
    if num_removed == 0:
        return df, splits, 0

    old_to_new: dict[int, int] = {}
    new_idx = 0
    for old_idx, is_valid in enumerate(valid_mask):
        if is_valid:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    filtered_df = df[valid_mask].reset_index(drop=True)
    filtered_splits: list[dict[str, list[int]]] = []
    for split in splits:
        new_split: dict[str, list[int]] = {}
        for key in ["train", "valid", "test"]:
            idxs = [int(i) for i in list(split.get(key, []))]
            new_split[key] = [old_to_new[i] for i in idxs if i in old_to_new]
        filtered_splits.append(new_split)

    return filtered_df, filtered_splits, num_removed


def _smiles_to_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(Descriptors._descList), np.nan, dtype=np.float32)

    values: list[float] = []
    for _, fn in Descriptors._descList:
        try:
            values.append(float(fn(mol)))
        except Exception:
            values.append(np.nan)
    return np.array(values, dtype=np.float32)


def _compute_descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    X = np.stack([_smiles_to_rdkit_descriptors(s) for s in smiles_list], axis=0)
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.clip(X, -1e6, 1e6)
    return X.astype(np.float32)


def run_chemprop_prepared_classification(
    datasets: str = "SARSCoV2_3CLPro_Diamond,ogbg-molhiv,hERG,DILI,AMES",
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    output_dir: Path = Path("repro_results/chemprop_prepared"),
    prep_cache_dir: Path = Path("repro_data/chemprop_prepared"),
    output_long_csv: Path = Path("repro_results/classification_results_chemprop_long.csv"),
    output_table_csv: Path = Path("repro_results/classification_results_chemprop_table.csv"),
    output_split_csv: Path = Path("repro_results/classification_results_chemprop_splits.csv"),
    merge_with_ml_long_csv: Path = Path("repro_results/classification_results_all_long.csv"),
    merged_long_csv: Path = Path("repro_results/classification_results_focus_with_chemprop_long.csv"),
    merged_table_csv: Path = Path("repro_results/classification_results_focus_with_chemprop_table.csv"),
    target_col: str = "",
    epochs: int = 30,
    patience: int = 10,
    seed: int = 42,
    skip_completed: int = 1,
    use_rdkit_descriptors: int = 0,
    model_name: str = "",
    score_scale: float = 100.0,
) -> None:
    """Run Chemprop for each dataset and each prepared split, then summarize AUROC."""
    dataset_list = [x.strip() for x in datasets.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_cache_dir.mkdir(parents=True, exist_ok=True)

    use_rdkit = bool(use_rdkit_descriptors)
    model_label = model_name.strip() if model_name.strip() else ("chemprop_rdkit" if use_rdkit else "chemprop")

    for dataset in dataset_list:
        print(f"=== Dataset: {dataset} ===")
        df_raw, _, splits = _load_prepared_dataset(
            dataset=dataset,
            prepared_dir=prepared_dir,
            prepared_src_dir=prepared_src_dir,
            prepared_extra_dir=prepared_extra_dir,
        )
        smiles_col, y_col = _select_smiles_target_columns(df_raw, target_col=target_col)
        df = (
            df_raw[[smiles_col, y_col]]
            .rename(columns={smiles_col: "smiles", y_col: "target"})
            .dropna()
            .reset_index(drop=True)
        )
        df["target"] = df["target"].astype(float)
        df, splits, num_removed = _filter_invalid_smiles_and_fix_splits(
            df=df, splits=splits, smiles_col="smiles"
        )
        if num_removed > 0:
            print(f"{dataset}: removed {num_removed} invalid SMILES before Chemprop training.")
        descriptor_matrix = None
        if use_rdkit:
            descriptor_matrix = _compute_descriptor_matrix(df["smiles"].astype(str).tolist())

        split_scores: list[float] = []
        for split_i, split in enumerate(splits, start=1):
            split_dir = prep_cache_dir / dataset / f"split_{split_i}"
            split_dir.mkdir(parents=True, exist_ok=True)

            split_labels = np.array(["unused"] * len(df), dtype=object)
            train_idx = np.array([int(i) for i in list(split["train"])], dtype=int)
            split_labels[train_idx] = "train"
            valid_idx = np.array([int(i) for i in list(split.get("valid", []))], dtype=int)
            if len(valid_idx) > 0:
                split_labels[valid_idx] = "val"
            test_idx = np.array([int(i) for i in list(split["test"])], dtype=int)
            split_labels[test_idx] = "test"

            data_csv = split_dir / "data.csv"
            data_df = df.copy()
            data_df["split"] = split_labels
            data_df.to_csv(data_csv, index=False)
            descriptors_npz = split_dir / "descriptors.npz"
            test_descriptors_npz = split_dir / "test_descriptors.npz"
            if use_rdkit and descriptor_matrix is not None:
                np.savez_compressed(descriptors_npz, descriptor_matrix)

            y_test = df.iloc[test_idx]["target"].to_numpy()
            test_input_csv = split_dir / "test_input.csv"
            df.iloc[test_idx][["smiles"]].to_csv(test_input_csv, index=False)
            if use_rdkit and descriptor_matrix is not None:
                np.savez_compressed(test_descriptors_npz, descriptor_matrix[test_idx])

            model_dir = output_dir / dataset / f"split_{split_i}"
            preds_csv = split_dir / "preds.csv"

            best_pt = model_dir / "model_0" / "best.pt"
            should_skip = bool(skip_completed)
            if not (should_skip and best_pt.exists() and preds_csv.exists()):
                train_cmd = [
                    "chemprop",
                    "train",
                    "-i",
                    str(data_csv),
                    "--task-type",
                    "classification",
                    "--splits-column",
                    "split",
                    "--smiles-columns",
                    "smiles",
                    "--target-columns",
                    "target",
                    "--metrics",
                    "roc",
                    "--tracking-metric",
                    "roc",
                    "--epochs",
                    str(epochs),
                    "--patience",
                    str(patience),
                    "--pytorch-seed",
                    str(seed),
                    "--data-seed",
                    str(seed),
                    "--accelerator",
                    "cpu",
                    "--save-dir",
                    str(model_dir),
                ]
                if use_rdkit:
                    train_cmd.extend(
                        [
                            "--descriptors-path",
                            str(descriptors_npz),
                            "--no-descriptor-scaling",
                        ]
                    )
                _run(train_cmd)

                predict_cmd = [
                    "chemprop",
                    "predict",
                    "-i",
                    str(test_input_csv),
                    "--smiles-columns",
                    "smiles",
                    "--model-paths",
                    str(model_dir),
                    "-o",
                    str(preds_csv),
                    "--drop-extra-columns",
                    "--accelerator",
                    "cpu",
                ]
                if use_rdkit:
                    predict_cmd.extend(
                        [
                            "--descriptors-path",
                            str(test_descriptors_npz),
                            "--no-descriptor-scaling",
                        ]
                    )
                _run(predict_cmd)
            else:
                print(f"Split {split_i}: reusing existing checkpoint/predictions.")

            preds = pd.read_csv(preds_csv)
            pred_col = _pick_prediction_column(preds_df=preds, smiles_col="smiles")
            y_score = preds[pred_col].to_numpy(dtype=float)

            if len(np.unique(y_test)) < 2:
                split_auc = float("nan")
                print(f"Split {split_i}: AUROC undefined (single class test split), marking NaN")
            else:
                split_auc = float(roc_auc_score(y_test, y_score))
            split_scores.append(split_auc)
            split_rows.append(
                {
                    "dataset": dataset,
                    "model": model_label,
                    "metric": "AUROC",
                    "split": split_i,
                    "score": split_auc,
                    "score_pct": split_auc * score_scale if np.isfinite(split_auc) else np.nan,
                    "seed": seed,
                    "epochs": epochs,
                    "patience": patience,
                }
            )
            print(f"Split {split_i}: AUROC = {split_auc}")

        scores = np.array(split_scores, dtype=float)
        valid_count = int(np.sum(np.isfinite(scores)))
        mean_auc = float(np.nanmean(scores))
        std_auc = float(np.nanstd(scores))
        rows.append(
            {
                "dataset": dataset,
                "model": model_label,
                "metric": "AUROC",
                "mean": mean_auc,
                "std": std_auc,
                "mean_std": f"{mean_auc:.4f} +/- {std_auc:.4f}",
                "mean_pct": mean_auc * score_scale,
                "std_pct": std_auc * score_scale,
                "mean_std_pct": f"{mean_auc * score_scale:.1f} +/- {std_auc * score_scale:.1f}",
                "seed": seed,
                "epochs": epochs,
                "patience": patience,
                "valid_splits": valid_count,
                "n_molecules": len(df),
            }
        )
        print(f"{dataset}: AUROC mean +/- std = {mean_auc:.4f} +/- {std_auc:.4f} (valid={valid_count})")

    chemprop_df = pd.DataFrame(rows).sort_values(["dataset"]).reset_index(drop=True)
    output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    chemprop_df.to_csv(output_long_csv, index=False)
    if split_rows:
        split_df = pd.DataFrame(split_rows).sort_values(["dataset", "split"]).reset_index(drop=True)
        output_split_csv.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(output_split_csv, index=False)
        print(f"Saved Chemprop split-level results to {output_split_csv}")

    chemprop_table = (
        chemprop_df.pivot(index="model", columns="dataset", values="mean_std_pct")
        .reset_index()
        .rename(columns={"model": "Model"})
    )
    chemprop_table.to_csv(output_table_csv, index=False)
    print(f"Saved Chemprop long results to {output_long_csv}")
    print(f"Saved Chemprop table results to {output_table_csv}")

    if merge_with_ml_long_csv.exists():
        ml_df = pd.read_csv(merge_with_ml_long_csv)
        ml_focus = ml_df[ml_df["dataset"].isin(dataset_list)].copy()
        merged = pd.concat([ml_focus, chemprop_df], ignore_index=True, sort=False)
        merged = merged.sort_values(["dataset", "model"]).reset_index(drop=True)
        merged.to_csv(merged_long_csv, index=False)

        merged_table = (
            merged.pivot(index="model", columns="dataset", values="mean_std_pct")
            .reset_index()
            .rename(columns={"model": "Model"})
        )
        merged_table.to_csv(merged_table_csv, index=False)
        print(f"Saved merged long results to {merged_long_csv}")
        print(f"Saved merged table results to {merged_table_csv}")
    else:
        print(f"Skip merge: {merge_with_ml_long_csv} does not exist.")


if __name__ == "__main__":
    from tap import tapify

    tapify(run_chemprop_prepared_classification)
