"""Create a SARS interpretability panel from Chemprop predictions."""

from pathlib import Path
import sys

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


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


def make_sars_interpretability_panel(
    preds_csv: Path = Path("repro_data/chemprop_prepared_focus_final/SARSCoV2_3CLPro_Diamond/split_1/preds.csv"),
    test_input_csv: Path = Path("repro_data/chemprop_prepared_focus_final/SARSCoV2_3CLPro_Diamond/split_1/test_input.csv"),
    dataset: str = "SARSCoV2_3CLPro_Diamond",
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    top_n: int = 12,
    out_dir: Path = Path("repro_results/plots/sars_interpretability"),
) -> None:
    prepared_src_dir = prepared_src_dir.resolve()
    if str(prepared_src_dir) not in sys.path:
        sys.path.insert(0, str(prepared_src_dir))

    data_path = _resolve_prepared_joblib(dataset, prepared_dir, prepared_extra_dir)
    obj = joblib.load(data_path)
    full_df = obj.data.copy()
    smiles_col, target_col = _select_smiles_target_columns(full_df)
    full_df = full_df[[smiles_col, target_col]].dropna().rename(columns={smiles_col: "smiles", target_col: "target"})
    full_df["target"] = full_df["target"].astype(float)

    preds = pd.read_csv(preds_csv)
    test_df = pd.read_csv(test_input_csv)
    pred_cols = [c for c in preds.columns if c != "smiles"]
    if not pred_cols:
        raise ValueError(f"No prediction column found in {preds_csv}")
    pred_col = pred_cols[0]

    merged = test_df.merge(preds[["smiles", pred_col]], on="smiles", how="left").rename(columns={pred_col: "pred"})
    merged = merged.merge(full_df, on="smiles", how="left")
    merged = merged.sort_values("pred", ascending=False).head(top_n).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"{dataset}_top_predicted_actives.csv", index=False)

    mols = []
    legends = []
    for _, row in merged.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"Pred={row['pred']:.3f} | Y={row['target']:.0f}" if pd.notna(row["target"]) else f"Pred={row['pred']:.3f}")

    if not mols:
        raise ValueError("No valid molecules to draw.")

    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(280, 220), legends=legends, useSVG=False)
    png_path = out_dir / f"{dataset}_top_predicted_panel.png"
    img.save(str(png_path))
    print(f"Saved {png_path}")
    print(f"Saved {out_dir / f'{dataset}_top_predicted_actives.csv'}")


if __name__ == "__main__":
    from tap import tapify

    tapify(make_sars_interpretability_panel)
