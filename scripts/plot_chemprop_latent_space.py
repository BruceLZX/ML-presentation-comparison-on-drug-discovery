"""Generate PCA/TSNE/UMAP plots from Chemprop fingerprints."""

import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return
    except FileNotFoundError:
        if cmd and cmd[0] == "chemprop":
            script_path = Path(sys.executable).parent / "chemprop"
            if script_path.exists():
                fallback = [str(script_path)] + cmd[1:]
                print(" ".join(fallback))
                subprocess.run(fallback, check=True)
                return
        raise


def _load_fingerprints(fp_path: Path) -> np.ndarray:
    if not fp_path.exists():
        candidates = sorted(fp_path.parent.glob(f"{fp_path.stem}_*.npz"))
        if not candidates:
            raise FileNotFoundError(fp_path)
        fp_path = candidates[0]

    data = np.load(fp_path, allow_pickle=True)
    if "arr_0" in data:
        return np.array(data["arr_0"], dtype=np.float32)
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"No arrays found in {fp_path}")
    return np.array(data[keys[0]], dtype=np.float32)


def _scatter_plot(points: np.ndarray, labels: np.ndarray, title: str, save_path: Path, method: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    classes = sorted(np.unique(labels))
    palette = {0: "#4C78A8", 1: "#E45756"}
    for c in classes:
        mask = labels == c
        label_name = "Inactive (0)" if int(c) == 0 else "Active (1)"
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=22,
            alpha=0.82,
            label=f"{label_name}, n={int(mask.sum())}",
            c=palette.get(int(c), "#444444"),
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel(f"{method} Component 1", fontsize=12)
    ax.set_ylabel(f"{method} Component 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d0d0", fontsize=10, title="Class")
    ax.grid(alpha=0.45, linestyle="--", linewidth=0.8, color="#D8D8D8")
    fig.tight_layout()
    fig.savefig(save_path, facecolor="white", transparent=False)
    plt.close(fig)
    with Image.open(save_path) as img:
        img.convert("RGB").save(save_path)
    print(f"Saved {save_path}")


def plot_chemprop_latent_space(
    dataset: str = "SARSCoV2_3CLPro_Diamond",
    model_dir: Path = Path("repro_results/chemprop_prepared_focus_final/SARSCoV2_3CLPro_Diamond/split_1"),
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    out_dir: Path = Path("repro_results/plots/chemprop_latent"),
    ffn_block_index: int = 0,
    seed: int = 42,
) -> None:
    prepared_src_dir = prepared_src_dir.resolve()
    if str(prepared_src_dir) not in sys.path:
        sys.path.insert(0, str(prepared_src_dir))

    dataset_path = _resolve_prepared_joblib(dataset, prepared_dir, prepared_extra_dir)
    obj = joblib.load(dataset_path)
    df = obj.data.copy()
    smiles_col, y_col = _select_smiles_target_columns(df)
    df = df[[smiles_col, y_col]].dropna().rename(columns={smiles_col: "smiles", y_col: "target"})
    df["target"] = df["target"].astype(float)
    df["target_bin"] = (df["target"] >= 0.5).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    input_csv = out_dir / f"{dataset}_fingerprint_input.csv"
    fp_npz = out_dir / f"{dataset}_fingerprints.npz"
    df[["smiles"]].to_csv(input_csv, index=False)

    _run(
        [
            "chemprop",
            "fingerprint",
            "-i",
            str(input_csv),
            "--smiles-columns",
            "smiles",
            "--model-paths",
            str(model_dir),
            "--ffn-block-index",
            str(ffn_block_index),
            "-o",
            str(fp_npz),
            "--accelerator",
            "cpu",
        ]
    )

    X = _load_fingerprints(fp_npz)
    y = df["target_bin"].to_numpy()
    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    pca_points = PCA(n_components=2, random_state=seed).fit_transform(X)
    _scatter_plot(
        pca_points,
        y,
        title=f"{dataset} | Chemprop Latent Space (PCA)",
        save_path=out_dir / f"{dataset}_chemprop_pca.png",
        method="PCA",
    )

    tsne_points = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=min(30, max(5, n // 50)),
    ).fit_transform(X)
    _scatter_plot(
        tsne_points,
        y,
        title=f"{dataset} | Chemprop Latent Space (t-SNE)",
        save_path=out_dir / f"{dataset}_chemprop_tsne.png",
        method="t-SNE",
    )

    try:
        import umap
    except ImportError:
        note_path = out_dir / f"{dataset}_umap_skipped.txt"
        note_path.write_text("UMAP not generated: umap-learn is not installed.\n")
        print(f"Saved {note_path}")
        return

    umap_points = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    _scatter_plot(
        umap_points,
        y,
        title=f"{dataset} | Chemprop Latent Space (UMAP)",
        save_path=out_dir / f"{dataset}_chemprop_umap.png",
        method="UMAP",
    )


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_chemprop_latent_space)
