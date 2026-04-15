"""Run RDKit descriptor feature selection on SARS using SHAP."""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _resolve_prepared_joblib(dataset: str, prepared_dir: Path, prepared_extra_dir: Path) -> Path:
    candidates = [
        prepared_dir / f"{dataset}.joblib",
        prepared_dir / "prepared" / f"{dataset}.joblib",
        prepared_extra_dir / f"{dataset}.joblib",
        prepared_extra_dir / "prepared" / f"{dataset}.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {dataset}.joblib in prepared dirs.")


def _select_smiles_target_columns(df: pd.DataFrame) -> tuple[str, str]:
    smiles_col = next((c for c in ["smiles", "SMILES", "Drug"] if c in df.columns), None)
    if smiles_col is None:
        raise ValueError("Could not find smiles column.")
    for c in ["Y", "target", "label", "labels"]:
        if c in df.columns:
            return smiles_col, c
    blocked = {"smiles", "SMILES", "Drug", "graph"}
    id_like = {c for c in df.columns if ("id" in c.lower() or "split" in c.lower())}
    candidates = [c for c in df.columns if c not in blocked and c not in id_like]
    if len(candidates) != 1:
        raise ValueError(f"Could not infer target column. Candidates: {candidates}")
    return smiles_col, candidates[0]


def _smiles_to_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(Descriptors._descList), np.nan, dtype=np.float32)
    vals = []
    for _, fn in Descriptors._descList:
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float32)


def _plot_ranked_barh(
    df: pd.DataFrame,
    value_col: str,
    feature_col: str,
    title: str,
    xlabel: str,
    save_path: Path,
    bar_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, max(6.2, len(df) * 0.33)), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.barh(
        df[feature_col],
        df[value_col],
        color=bar_color,
        edgecolor="#2f2f2f",
        linewidth=0.25,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("RDKit Descriptor", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#D8D8D8", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, facecolor="white", transparent=False)
    plt.close(fig)
    with Image.open(save_path) as img:
        img.convert("RGB").save(save_path)


def run_sars_rdkit_feature_selection(
    dataset: str = "SARSCoV2_3CLPro_Diamond",
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    top_n: int = 20,
    seed: int = 42,
    out_dir: Path = Path("repro_results/plots/sars_feature_selection"),
) -> None:
    prepared_src_dir = prepared_src_dir.resolve()
    if str(prepared_src_dir) not in sys.path:
        sys.path.insert(0, str(prepared_src_dir))

    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = _resolve_prepared_joblib(dataset, prepared_dir, prepared_extra_dir)
    obj = joblib.load(data_path)
    df = obj.data.copy()
    smiles_col, y_col = _select_smiles_target_columns(df)
    df = (
        df[[smiles_col, y_col]]
        .dropna()
        .rename(columns={smiles_col: "smiles", y_col: "target"})
        .reset_index(drop=True)
    )
    df["target"] = (df["target"].astype(float) >= 0.5).astype(int)

    desc_names = [name for name, _ in Descriptors._descList]
    X = np.stack([_smiles_to_rdkit_descriptors(s) for s in df["smiles"].astype(str).tolist()], axis=0)
    y = df["target"].to_numpy()

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=-1)),
        ]
    )
    pipe.fit(X, y)
    model = pipe.named_steps["model"]
    importances = model.feature_importances_

    rank_df = pd.DataFrame({"feature": desc_names, "importance": importances}).sort_values("importance", ascending=False)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    rank_df.to_csv(out_dir / f"{dataset}_rdkit_feature_importance.csv", index=False)

    top_df = rank_df.head(top_n).iloc[::-1]
    _plot_ranked_barh(
        df=top_df,
        value_col="importance",
        feature_col="feature",
        title=f"{dataset}: Top {top_n} RDKit Features (RandomForest)",
        xlabel="RandomForest Feature Importance",
        save_path=out_dir / f"{dataset}_top{top_n}_rdkit_feature_importance.png",
        bar_color="#2A9D8F",
    )

    try:
        import shap

        imp = pipe.named_steps["imputer"]
        sc = pipe.named_steps["scaler"]
        X_proc = sc.transform(imp.transform(X))
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_proc)
        if isinstance(shap_vals, list):
            # Binary classification often returns [class0, class1]
            shap_matrix = np.array(shap_vals[-1])
        else:
            shap_matrix = np.array(shap_vals)
            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, -1]
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        shap_df = pd.DataFrame({"feature": desc_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_df["rank"] = np.arange(1, len(shap_df) + 1)
        shap_df.to_csv(out_dir / f"{dataset}_rdkit_shap_importance.csv", index=False)

        top_shap = shap_df.head(top_n).iloc[::-1]
        _plot_ranked_barh(
            df=top_shap,
            value_col="mean_abs_shap",
            feature_col="feature",
            title=f"{dataset}: Top {top_n} RDKit Features (SHAP)",
            xlabel="Mean |SHAP value|",
            save_path=out_dir / f"{dataset}_top{top_n}_rdkit_shap_importance.png",
            bar_color="#F4A261",
        )
    except Exception as exc:
        (out_dir / f"{dataset}_shap_skipped.txt").write_text(f"SHAP step skipped: {exc}\n")

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    from tap import tapify

    tapify(run_sars_rdkit_feature_selection)
