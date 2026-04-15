"""Run Chemprop MCTS rationale extraction for top SARS candidate molecules."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Callable, Iterable

from lightning import pytorch as pl
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import torch

from chemprop import data, models
from chemprop.models import MPNN


def make_prediction(models_list: list[MPNN], trainer: pl.Trainer, smiles: list[str]) -> np.ndarray:
    test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
    test_dset = data.MoleculeDataset(test_data)
    test_loader = data.build_dataloader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    with torch.inference_mode():
        sum_preds = []
        for model in models_list:
            predss = trainer.predict(model, test_loader)
            preds = torch.cat(predss, 0)
            preds = preds.cpu().numpy()
            sum_preds.append(preds)
        sum_preds = sum(sum_preds)
        avg_preds = sum_preds / len(models_list)
    return avg_preds


@dataclass
class MCTSNode:
    smiles: str
    atoms: Iterable[int]
    W: float = 0.0
    N: int = 0
    P: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.atoms = set(self.atoms)

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int, c_puct: float = 10.0) -> float:
        return c_puct * self.P * math.sqrt(n) / (1 + self.N)


def find_clusters(mol: Chem.Mol) -> tuple[list[tuple[int, ...]], list[list[int]]]:
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [(0,)], [[0]]

    clusters: list[tuple[int, ...]] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def extract_subgraph_from_mol(mol: Chem.Mol, selected_atoms: set[int]) -> tuple[Chem.Mol, list[int]]:
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if bad_neis:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms
        ]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    for atom in sorted(remove_atoms, reverse=True):
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def extract_subgraph(smiles: str, selected_atoms: set[int]) -> tuple[str | None, list[int] | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    try:
        Chem.Kekulize(mol)
    except Exception:
        pass

    subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
    try:
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)
    except Exception:
        subgraph = None

    mol = Chem.MolFromSmiles(smiles)
    if subgraph is not None and mol is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    if mol is None:
        return None, None
    subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)
    if subgraph is None:
        return None, None
    return Chem.MolToSmiles(subgraph), roots


def mcts_rollout(
    node: MCTSNode,
    state_map: dict[str, MCTSNode],
    orig_smiles: str,
    clusters: list[set[int]],
    atom_cls: list[set[int]],
    nei_cls: list[set[int]],
    scoring_function: Callable[[list[str]], list[float]],
    min_atoms: int = 15,
    c_puct: float = 10.0,
) -> float:
    cur_atoms = node.atoms
    if len(cur_atoms) <= min_atoms:
        return node.P

    if len(node.children) == 0:
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or (len(clusters[i]) == 2 and len(leaf_atoms) == 1):
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = float(score)

    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count, c_puct=c_puct))
    v = mcts_rollout(
        selected_node,
        state_map,
        orig_smiles,
        clusters,
        atom_cls,
        nei_cls,
        scoring_function,
        min_atoms=min_atoms,
        c_puct=c_puct,
    )
    selected_node.W += v
    selected_node.N += 1
    return v


def mcts(
    smiles: str,
    scoring_function: Callable[[list[str]], list[float]],
    n_rollout: int,
    max_atoms: int,
    prop_delta: float,
    min_atoms: int = 15,
    c_puct: int = 10,
) -> list[MCTSNode]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(
            root,
            state_map,
            smiles,
            clusters,
            atom_cls,
            nei_cls,
            scoring_function,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

    rationales = [node for _, node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]
    return rationales


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


def _build_clean_mcts_grid(
    molecules: list[Chem.Mol],
    highlights: list[list[int]],
    legends: list[str],
    out_path: Path,
    mols_per_row: int = 2,
) -> None:
    if not molecules:
        return

    cell_w = 460
    cell_h = 340
    pad_x = 24
    pad_y = 26
    text_h = 38
    header_h = 64
    n_rows = int(math.ceil(len(molecules) / mols_per_row))
    canvas_w = pad_x + mols_per_row * cell_w + (mols_per_row - 1) * pad_x + pad_x
    canvas_h = header_h + pad_y + n_rows * (cell_h + text_h) + (n_rows - 1) * pad_y + pad_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    draw.text((pad_x, 18), "SARS MCTS Rationales (Top Predicted Molecules)", fill=(20, 20, 20), font=title_font)
    draw.line((pad_x, header_h - 8, canvas_w - pad_x, header_h - 8), fill=(205, 205, 205), width=2)

    for idx, mol in enumerate(molecules):
        row = idx // mols_per_row
        col = idx % mols_per_row
        x0 = pad_x + col * (cell_w + pad_x)
        y0 = header_h + pad_y + row * (cell_h + text_h + pad_y)

        img = Draw.MolToImage(
            mol,
            size=(cell_w, cell_h),
            highlightAtoms=highlights[idx],
            fitImage=True,
        )
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

        cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
        cell.paste(img, (0, 0))
        canvas.paste(cell, (x0, y0))
        draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(190, 190, 190), width=2)
        draw.text((x0 + 8, y0 + cell_h + 9), legends[idx], fill=(30, 30, 30), font=body_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


def run_sars_mcts_interpretability(
    dataset: str = "SARSCoV2_3CLPro_Diamond",
    model_path: Path = Path("repro_results/chemprop_prepared_focus7_plain/SARSCoV2_3CLPro_Diamond/split_1/model_0/best.pt"),
    candidate_csv: Path = Path("repro_results/plots/sars_interpretability/SARSCoV2_3CLPro_Diamond_top_predicted_actives.csv"),
    prepared_dir: Path = Path("external_data/prepared_all/prepared"),
    prepared_extra_dir: Path = Path("external_data/prepared_all/prepared_6addedfreesolv_ignorethese"),
    prepared_src_dir: Path = Path("external_data/src_all"),
    top_k: int = 4,
    n_rollout: int = 20,
    max_atoms: int = 20,
    min_atoms: int = 8,
    prop_delta: float = 0.5,
    c_puct: float = 10.0,
    out_dir: Path = Path("repro_results/plots/sars_mcts_interpretability"),
) -> None:
    prepared_src_dir = prepared_src_dir.resolve()
    if str(prepared_src_dir) not in sys.path:
        sys.path.insert(0, str(prepared_src_dir))

    out_dir.mkdir(parents=True, exist_ok=True)

    model = models.MPNN.load_from_file(model_path)
    trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="cpu", devices=1)

    data_path = _resolve_prepared_joblib(dataset, prepared_dir, prepared_extra_dir)
    obj = joblib.load(data_path)
    df = obj.data.copy()
    smiles_col, y_col = _select_smiles_target_columns(df)
    gt = (
        df[[smiles_col, y_col]]
        .dropna()
        .rename(columns={smiles_col: "smiles", y_col: "target"})
        .drop_duplicates("smiles")
    )
    gt["target"] = gt["target"].astype(float)

    candidates = pd.read_csv(candidate_csv).dropna(subset=["smiles"]).head(top_k).copy()

    def scoring_function(smiles_list: list[str]) -> list[float]:
        preds = make_prediction(models_list=[model], trainer=trainer, smiles=smiles_list)
        preds = np.array(preds).reshape(len(smiles_list), -1)
        return preds[:, 0].astype(float).tolist()

    rows: list[dict[str, object]] = []
    drawn_mols = []
    legends = []
    highlight_atom_lists: list[list[int]] = []

    for _, row in candidates.iterrows():
        smiles = str(row["smiles"])
        pred = float(row["pred"]) if "pred" in row and pd.notna(row["pred"]) else np.nan
        y = float(row["target"]) if "target" in row and pd.notna(row["target"]) else np.nan

        rationale_nodes = mcts(
            smiles=smiles,
            scoring_function=scoring_function,
            n_rollout=n_rollout,
            max_atoms=max_atoms,
            prop_delta=prop_delta,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

        if rationale_nodes:
            best = max(rationale_nodes, key=lambda n: n.P)
            rationale_smiles = best.smiles
            rationale_score = float(best.P)
            rationale_atoms = sorted(list(best.atoms))
        else:
            best = None
            rationale_smiles = ""
            rationale_score = np.nan
            rationale_atoms = []

        rows.append(
            {
                "smiles": smiles,
                "pred": pred,
                "target": y,
                "rationale_smiles": rationale_smiles,
                "rationale_score": rationale_score,
                "rationale_num_atoms": len(rationale_atoms),
            }
        )

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            drawn_mols.append(mol)
            highlight_atom_lists.append(rationale_atoms)
            legends.append(
                f"Pred={pred:.3f} | MCTS={rationale_score:.3f}" if not np.isnan(rationale_score) else f"Pred={pred:.3f} | MCTS=n/a"
            )

            img = Draw.MolToImage(mol, size=(600, 450), highlightAtoms=rationale_atoms)
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
            safe_smi = smiles.replace("/", "_").replace("\\", "_").replace(":", "_")
            img.save(str(out_dir / f"mcts_{safe_smi[:40]}.png"))

    out_csv = out_dir / f"{dataset}_mcts_rationales.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if drawn_mols:
        grid_path = out_dir / f"{dataset}_mcts_rationale_grid.png"
        _build_clean_mcts_grid(
            molecules=drawn_mols,
            highlights=highlight_atom_lists,
            legends=legends,
            out_path=grid_path,
            mols_per_row=2,
        )
        print(f"Saved {grid_path}")

    print(f"Saved {out_csv}")


if __name__ == "__main__":
    from tap import tapify

    tapify(run_sars_mcts_interpretability)
