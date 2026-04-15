"""Train/evaluate sklearn baselines on TDC or prepared joblib datasets."""

from pathlib import Path
import random
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tdc.single_pred import ADME, Tox


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bitvect = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def _smiles_to_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


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


def _build_classifier(classifier: str, seed: int) -> tuple[Pipeline, dict[str, list[object]]]:
    if classifier == "logistic_regression":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, n_jobs=-1, random_state=seed)),
            ]
        )
        param_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        }
        return model, param_grid

    if classifier == "random_forest":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
            ]
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 5],
        }
        return model, param_grid

    if classifier == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost` and rerun."
            ) from exc

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        n_jobs=-1,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
        }
        return model, param_grid

    if classifier == "gradient_boosting":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=seed)),
            ]
        )
        param_grid = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
        return model, param_grid

    if classifier == "extra_trees":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
            ]
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 5],
        }
        return model, param_grid

    if classifier == "ada_boost":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", AdaBoostClassifier(random_state=seed)),
            ]
        )
        param_grid = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1, 0.3],
        }
        return model, param_grid

    raise ValueError(
        f"Unsupported classifier: {classifier}. "
        "Use 'logistic_regression', 'random_forest', 'gradient_boosting', "
        "'extra_trees', 'ada_boost', or 'xgboost'."
    )


def _build_regressor(regressor: str, seed: int) -> tuple[Pipeline, dict[str, list[object]]]:
    if regressor == "ridge":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(random_state=seed)),
            ]
        )
        param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
        return model, param_grid

    if regressor == "random_forest":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)),
            ]
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 5],
        }
        return model, param_grid

    if regressor == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost` and rerun."
            ) from exc

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        n_jobs=-1,
                        eval_metric="mae",
                    ),
                ),
            ]
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
        }
        return model, param_grid

    if regressor == "gradient_boosting":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=seed)),
            ]
        )
        param_grid = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
        return model, param_grid

    if regressor == "extra_trees":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1)),
            ]
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 5],
        }
        return model, param_grid

    if regressor == "ada_boost":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", AdaBoostRegressor(random_state=seed)),
            ]
        )
        param_grid = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1, 0.3],
        }
        return model, param_grid

    raise ValueError(
        f"Unsupported regressor: {regressor}. "
        "Use 'ridge', 'random_forest', 'gradient_boosting', 'extra_trees', 'ada_boost', or 'xgboost'."
    )


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
        + ", ".join(str(p.parent) for p in candidates)
    )


def _load_prepared_dataset(
    dataset: str,
    prepared_dir: Path,
    prepared_src_dir: Path,
    prepared_extra_dir: Path | None = None,
) -> tuple[pd.DataFrame, str, list[dict[str, list[int]]], str]:
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

    df = obj.data.copy()
    splits = obj.splits
    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError(f"Expected list of split dicts in {dataset_path}, got {type(splits)}")

    return df, str(obj.task), splits, "prepared_joblib"


def _load_tdc_dataset(dataset: str, save_dir: Path, n_splits: int) -> tuple[pd.DataFrame, str, list[dict[str, list[int]]], str]:
    try:
        data_obj = ADME(name=dataset, path=save_dir)
        df = data_obj.get_data()
        source = "ADME"
    except Exception:
        data_obj = Tox(name=dataset, path=save_dir)
        df = data_obj.get_data()
        source = "Tox"

    if "Drug" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"TDC dataset {dataset} missing required columns Drug and Y.")

    df = df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "target"}).dropna().reset_index(drop=True)
    scaffolds = np.array([_smiles_to_scaffold(s) for s in df["smiles"].tolist()])
    y_tmp = df["target"].astype(float).to_numpy()

    splits: list[dict[str, list[int]]] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(df.index.to_numpy(), y_tmp, groups=scaffolds):
        splits.append({"train": train_idx.tolist(), "valid": [], "test": test_idx.tolist()})

    inferred_task = _infer_task(y_tmp)
    return df, inferred_task, splits, source


def _infer_task(y: np.ndarray) -> str:
    unique_targets = np.unique(y)
    is_classification = len(unique_targets) == 2 and set(unique_targets).issubset({0.0, 1.0})
    return "classification" if is_classification else "regression"


def _select_smiles_target_columns(df: pd.DataFrame, target_col: str = "") -> tuple[str, str]:
    smiles_candidates = ["smiles", "SMILES", "Drug"]
    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError(f"Could not find smiles column. Available columns: {list(df.columns)}")

    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"target_col={target_col} not in dataframe columns.")
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


def _compute_features(
    smiles_list: list[str],
    feature_type: str,
    cache_path: Path,
    use_feature_cache: bool,
) -> np.ndarray:
    cache_loaded = False
    X = None

    if use_feature_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_smiles = cached["smiles"]
        current_smiles = np.array(smiles_list, dtype=str)
        if len(cached_smiles) == len(current_smiles) and np.array_equal(cached_smiles, current_smiles):
            X = cached["X"]
            cache_loaded = True
            print(f"Loaded cached features from {cache_path}")

    if X is None and feature_type == "morgan":
        X = np.stack([_smiles_to_morgan_fp(s) for s in smiles_list])
    elif X is None and feature_type == "rdkit_descriptors":
        X = np.stack([_smiles_to_rdkit_descriptors(s) for s in smiles_list])
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.clip(X, -1e6, 1e6)
    elif X is None:
        raise ValueError(f"Unsupported feature_type: {feature_type}. Use 'morgan' or 'rdkit_descriptors'.")

    if use_feature_cache and not cache_loaded:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X.astype(np.float32), smiles=np.array(smiles_list, dtype=str))
        print(f"Saved feature cache to {cache_path}")

    return X


def run_tdc_single_scaffold_cv_baseline(
    dataset: str = "BBB_Martins",
    dataset_source: str = "prepared",
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    save_dir: Path = Path("repro_data/tdc_single"),
    n_splits: int = 5,
    target_col: str = "",
    feature_type: str = "rdkit_descriptors",
    classifier: str = "logistic_regression",
    regressor: str = "ridge",
    seed: int = 42,
    use_feature_cache: bool = True,
    do_grid_search: bool = False,
    grid_search_inner_folds: int = 3,
) -> dict[str, object]:
    """Run scaffold-based CV baseline on a single dataset.

    dataset_source:
    - prepared: load mentor-provided Dataset object via joblib (uses fixed 3x train/val/test splits)
    - tdc: load directly from TDC and build GroupKFold scaffold splits
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    _set_seed(seed)

    if dataset_source == "prepared":
        df_raw, task, splits, source = _load_prepared_dataset(
            dataset=dataset,
            prepared_dir=prepared_dir,
            prepared_src_dir=prepared_src_dir,
            prepared_extra_dir=prepared_extra_dir,
        )
    elif dataset_source == "tdc":
        df_raw, task, splits, source = _load_tdc_dataset(dataset=dataset, save_dir=save_dir, n_splits=n_splits)
    else:
        raise ValueError("dataset_source must be 'prepared' or 'tdc'.")

    smiles_col, y_col = _select_smiles_target_columns(df_raw, target_col=target_col)
    df = df_raw[[smiles_col, y_col]].rename(columns={smiles_col: "smiles", y_col: "target"}).dropna().reset_index(drop=True)

    y = df["target"].astype(float).to_numpy()
    inferred_task = _infer_task(y)
    if task not in {"classification", "regression"}:
        task = inferred_task
    if task != inferred_task:
        print(f"Warning: dataset task={task} but inferred from labels={inferred_task}. Using dataset task.")

    cache_key = f"{dataset_source}_{dataset}_{feature_type}".lower()
    cache_path = save_dir / "feature_cache" / f"{cache_key}.npz"
    smiles = df["smiles"].astype(str).tolist()
    X = _compute_features(smiles_list=smiles, feature_type=feature_type, cache_path=cache_path, use_feature_cache=use_feature_cache)
    scaffolds = np.array([_smiles_to_scaffold(s) for s in smiles])

    fold_scores: list[float] = []
    tuning_scores: list[float] = []
    best_params_by_fold: list[dict[str, object]] = []
    overall_best_params = None
    overall_best_model = None

    for split_i, split in enumerate(splits, start=1):
        train_idx = np.array(split["train"], dtype=int)
        valid_idx = np.array(split.get("valid", []), dtype=int)
        test_idx = np.array(split["test"], dtype=int)
        use_valid = len(valid_idx) > 0

        if use_valid:
            fit_idx = np.concatenate([train_idx, valid_idx])
        else:
            fit_idx = train_idx

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if task == "classification":
            model, param_grid = _build_classifier(classifier=classifier, seed=seed)
            scoring = "roc_auc"
            metric_name = "AUROC"
        else:
            model, param_grid = _build_regressor(regressor=regressor, seed=seed)
            scoring = "neg_mean_absolute_error"
            metric_name = "MAE"

        if do_grid_search:
            if use_valid:
                test_fold = np.array([-1] * len(train_idx) + [0] * len(valid_idx))
                inner_cv = PredefinedSplit(test_fold=test_fold)
                search_X = np.concatenate([X[train_idx], X[valid_idx]], axis=0)
                search_y = np.concatenate([y[train_idx], y[valid_idx]], axis=0)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=inner_cv, n_jobs=-1, refit=True)
                search.fit(search_X, search_y)
            else:
                inner_groups = scaffolds[fit_idx]
                inner_n_splits = min(grid_search_inner_folds, len(np.unique(inner_groups)))
                if inner_n_splits < 2:
                    raise ValueError("Not enough unique scaffolds in training fold for grid search CV.")
                search = GridSearchCV(
                    model,
                    param_grid,
                    scoring=scoring,
                    cv=GroupKFold(n_splits=inner_n_splits),
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_fit, y_fit, groups=inner_groups)

            model = search.best_estimator_
            best_params = dict(search.best_params_)
            best_params_by_fold.append(best_params)
            tuning_scores.append(float(search.best_score_))
            print(f"Split {split_i}: best params = {best_params}")
        else:
            model.fit(X_fit, y_fit)

        if task == "classification":
            y_score = model.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) < 2:
                score = float("nan")
                print(f"Split {split_i}: AUROC undefined (single class in test), marking as NaN")
            else:
                score = roc_auc_score(y_test, y_score)
        else:
            y_pred = model.predict(X_test)
            score = mean_absolute_error(y_test, y_pred)

        fold_scores.append(float(score))
        print(f"Split {split_i}: {metric_name} = {score:.4f}")

    if do_grid_search and best_params_by_fold:
        pick_idx = int(np.argmax(tuning_scores))
        overall_best_params = best_params_by_fold[pick_idx]
        if task == "classification":
            overall_best_model, _ = _build_classifier(classifier=classifier, seed=seed)
        else:
            overall_best_model, _ = _build_regressor(regressor=regressor, seed=seed)
        overall_best_model.set_params(**overall_best_params)
        overall_best_model.fit(X, y)

    print()
    print(f"Dataset: {dataset} ({source})")
    print(f"Task type: {task}")
    if task == "classification":
        print(f"Classifier: {classifier}")
    else:
        print(f"Regressor: {regressor}")
    print(f"Features: {feature_type} ({X.shape[1]} columns)")
    print(f"Seed: {seed}")
    valid_scores = np.array(fold_scores, dtype=float)
    valid_count = int(np.sum(np.isfinite(valid_scores)))
    mean_score = float(np.nanmean(valid_scores))
    std_score = float(np.nanstd(valid_scores))
    print(f"{metric_name} mean ± std over {len(splits)} scaffold splits (valid={valid_count}): {mean_score:.4f} ± {std_score:.4f}")
    if do_grid_search and overall_best_params is not None:
        print(f"Best params selected from CV: {overall_best_params}")

    return {
        "dataset": dataset,
        "source": source,
        "task_type": task,
        "metric_name": metric_name,
        "mean": mean_score,
        "std": std_score,
        "valid_splits": valid_count,
        "fold_scores": fold_scores,
        "n_splits": len(splits),
        "n_molecules": len(df),
        "n_features": int(X.shape[1]),
        "best_model": overall_best_model,
        "best_params": overall_best_params,
    }


if __name__ == "__main__":
    from tap import tapify

    tapify(run_tdc_single_scaffold_cv_baseline)
