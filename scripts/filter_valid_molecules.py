"""Filter CSV files in a directory to keep only rows with valid RDKit SMILES."""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from tap import tapify
from tqdm import tqdm


def filter_valid_smiles(
    data_dir: Path,
    smiles_column: str = "smiles",
) -> None:
    """For each CSV in the directory, drop rows whose SMILES are invalid (RDKit MolFromSmiles is None) and overwrite the file.

    :param data_dir: Directory containing CSV files to filter.
    :param smiles_column: Name of the column containing SMILES strings.
    """
    for csv_path in tqdm(sorted(data_dir.glob("**/*.csv"))):
        df = pd.read_csv(csv_path)

        original_len = len(df)
        mols = df[smiles_column].apply(Chem.MolFromSmiles)
        valid = mols.notnull()
        df_filtered = df.loc[valid].reset_index(drop=True)
        removed = original_len - len(df_filtered)

        if removed > 0:
            df_filtered.to_csv(csv_path, index=False)
            print(
                f"{csv_path.name}: kept {len(df_filtered):,} / {original_len:,} rows (removed {removed:,} invalid SMILES)"
            )
        else:
            print(f"{csv_path.name}: all {original_len:,} rows valid, no change.")


if __name__ == "__main__":
    tapify(filter_valid_smiles)
