"""Generate RDKit descriptor .npz files for prepared TDC ADMET group datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from tdc_constants import ADMET_GROUP_SMILES_COLUMN


def _compute_rdkit_descriptors(smiles: str) -> np.ndarray:
    """Compute a vector of RDKit scalar descriptors for one SMILES."""
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


def prepare_rdkit_descriptors_group(data_dir: Path) -> None:
    """Create data.npz descriptor files next to each seed's data.csv.

    :param data_dir: Directory containing dataset/seed/data.csv files.
    """
    dataset_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    for dataset_dir in tqdm(dataset_dirs, desc="datasets"):
        seed_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        for seed_dir in tqdm(seed_dirs, desc=dataset_dir.name, leave=False):
            csv_path = seed_dir / "data.csv"
            npz_path = seed_dir / "data.npz"

            data = pd.read_csv(csv_path)
            descs = np.stack([_compute_rdkit_descriptors(s) for s in data[ADMET_GROUP_SMILES_COLUMN].tolist()])
            descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)

            # Chemprop expects descriptors at key "arr_0" in .npz
            np.savez_compressed(npz_path, descs)


if __name__ == "__main__":
    from tap import tapify

    tapify(prepare_rdkit_descriptors_group)
