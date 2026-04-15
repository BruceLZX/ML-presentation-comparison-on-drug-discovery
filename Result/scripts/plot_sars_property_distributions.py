"""Create readable SARS property-distribution plots with robust x-axis limits."""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def _resolve_prepared_joblib(
    dataset: str,
    prepared_dir: Path,
    prepared_extra_dir: Path,
) -> Path:
    candidates = [
        prepared_dir / f"{dataset}.joblib",
        prepared_dir / "prepared" / f"{dataset}.joblib",
        prepared_extra_dir / f"{dataset}.joblib",
        prepared_extra_dir / "prepared" / f"{dataset}.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {dataset}.joblib in provided prepared directories.")


def _select_smiles_target_columns(df: pd.DataFrame) -> tuple[str, str]:
    smiles_candidates = ["smiles", "SMILES", "Drug"]
    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError(f"Could not find smiles column in dataframe columns: {list(df.columns)}")

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
            f"Candidate columns: {candidates}"
        )
    return smiles_col, candidates[0]


def _compute_rdkit_properties(smiles: str) -> tuple[float, float, float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return (
        float(Descriptors.MolWt(mol)),
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.TPSA(mol)),
    )


def _robust_limits(values: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    lo = float(np.quantile(vals, q_low))
    hi = float(np.quantile(vals, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _plot_one_property(
    prop_df: pd.DataFrame,
    dataset: str,
    prop_col: str,
    display_name: str,
    xlabel: str,
    units: str,
    out_dir: Path,
) -> None:
    inactive = prop_df.loc[prop_df["Y"] < 0.5, prop_col].astype(float).to_numpy()
    active = prop_df.loc[prop_df["Y"] >= 0.5, prop_col].astype(float).to_numpy()
    both = np.concatenate([inactive, active], axis=0)
    x_min, x_max = _robust_limits(both, q_low=0.01, q_high=0.99)

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bins = 40
    ax.hist(
        inactive,
        bins=bins,
        range=(x_min, x_max),
        density=True,
        alpha=0.52,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.35,
        label=f"Inactive (n={len(inactive)})",
    )
    ax.hist(
        active,
        bins=bins,
        range=(x_min, x_max),
        density=True,
        alpha=0.52,
        color="#E45756",
        edgecolor="white",
        linewidth=0.35,
        label=f"Active (n={len(active)})",
    )

    outlier_cnt = int(np.sum((both < x_min) | (both > x_max)))
    total = max(1, len(both))
    outlier_pct = 100.0 * outlier_cnt / total
    short_name = dataset.replace("_", " ")
    ax.set_title(f"{short_name} | {display_name} Distribution", fontsize=16, fontweight="bold", pad=12)
    label = f"{xlabel} ({units})" if units else xlabel
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(alpha=0.45, linestyle="--", linewidth=0.8, color="#D8D8D8")
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d0d0", fontsize=10, title="Class")
    ax.text(
        0.01,
        0.985,
        f"x-axis clipped to 1st-99th percentile; hidden outliers: {outlier_cnt}/{total} ({outlier_pct:.1f}%)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout()
    png_path = out_dir / f"{dataset}_{prop_col}_distribution.png"
    pdf_path = out_dir / f"{dataset}_{prop_col}_distribution.pdf"
    fig.savefig(png_path, facecolor="white", transparent=False)
    fig.savefig(pdf_path, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_sars_property_distributions(
    dataset: str = "SARSCoV2_3CLPro_Diamond",
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    out_dir: Path = Path("repro_results/plots/sars_property_distributions"),
) -> None:
    """Build 3 readable RDKit property distribution plots for SARS dataset molecules."""

    prepared_src_dir = prepared_src_dir.resolve()
    if str(prepared_src_dir) not in sys.path:
        sys.path.insert(0, str(prepared_src_dir))

    data_path = _resolve_prepared_joblib(
        dataset=dataset,
        prepared_dir=prepared_dir,
        prepared_extra_dir=prepared_extra_dir,
    )
    obj = joblib.load(data_path)

    df = obj.data.copy()
    smiles_col, target_col = _select_smiles_target_columns(df)
    df = df[[smiles_col, target_col]].dropna().rename(columns={smiles_col: "smiles", target_col: "Y"})
    df["Y"] = df["Y"].astype(float)

    rows: list[dict[str, float | str]] = []
    for smiles, y in zip(df["smiles"], df["Y"]):
        props = _compute_rdkit_properties(smiles)
        if props is None:
            continue
        mw, logp, tpsa = props
        rows.append({"smiles": smiles, "Y": y, "MolWt": mw, "MolLogP": logp, "TPSA": tpsa})

    prop_df = pd.DataFrame(rows)
    if prop_df.empty:
        raise ValueError("No valid molecules after RDKit parsing.")

    out_dir.mkdir(parents=True, exist_ok=True)
    active_csv = out_dir / f"{dataset}_active.csv"
    inactive_csv = out_dir / f"{dataset}_inactive.csv"
    prop_df[prop_df["Y"] >= 0.5].to_csv(active_csv, index=False)
    prop_df[prop_df["Y"] < 0.5].to_csv(inactive_csv, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "grid.color": "#D8D8D8",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    for property_column, display_name, xlabel, units in [
        ("MolWt", "Molecular Weight", "Molecular Weight", "Da"),
        ("MolLogP", "LogP", "LogP", ""),
        ("TPSA", "TPSA", "Topological Polar Surface Area", "A^2"),
    ]:
        _plot_one_property(
            prop_df=prop_df,
            dataset=dataset,
            prop_col=property_column,
            display_name=display_name,
            xlabel=xlabel,
            units=units,
            out_dir=out_dir,
        )

    print(f"Saved per-class CSVs to {out_dir}")


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_sars_property_distributions)
