"""Tests for ADMET-AI."""

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from admet_ai import admet_predict, ADMETModel
from admet_ai.constants import DEFAULT_ADMET_PATH, DEFAULT_DRUGBANK_PATH

# Tolerances for comparing PyTorch predictions across Python versions and operating systems.
RTOL = 1e-3
ATOL = 1e-4

# Percentiles are computed by scipy.stats.percentileofscore (rank-based). They are a *step function*
# of the prediction: a tiny change in the prediction can cross reference values and jump the
# percentile. So we use looser tolerances for percentile columns to avoid flaky CI across OS/Python.
PERCENTILE_RTOL = 0.01  # 1% relative
PERCENTILE_ATOL = 1.0  # 1 percentile point


ADMET_DATA = pd.read_csv(DEFAULT_ADMET_PATH)
DRUGBANK_DATA = pd.read_csv(DEFAULT_DRUGBANK_PATH)

FIRST_DRUGBANK_ROW = DRUGBANK_DATA.iloc[0]
FIRST_DRUGBANK_SMILES = FIRST_DRUGBANK_ROW["smiles"]

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
DRUGBANK_DATA_PERCENTILES = pd.read_csv(TEST_DATA_DIR / "drugbank_approved_percentiles.csv.bz2")
DRUGBANK_DATA_PERCENTILES_NERVOUS_SYSTEM = pd.read_csv(
    TEST_DATA_DIR / "drugbank_approved_percentiles_nervous_system.csv.bz2"
)


def _is_percentile_column(column: str) -> bool:
    """True if the column holds DrugBank percentile values.

    :param column: The column name.
    :return: True if the column holds DrugBank percentile values.
    """
    return column.endswith("_percentile")


class TestADMETPredict:
    @pytest.mark.parametrize("include_physchem", [True, False])
    def test_admet_predict_single_smiles(self, include_physchem: bool) -> None:
        """Test admet_predict with the first DrugBank SMILES."""
        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            preds_path = Path(temp_dir) / "preds.csv"

            pd.DataFrame({"smiles": [FIRST_DRUGBANK_SMILES]}).to_csv(data_path, index=False)

            admet_predict(
                data_path=data_path,
                save_path=preds_path,
                include_physchem=include_physchem,
                drugbank_path=None,
                atc_code=None,
            )

            preds = pd.read_csv(preds_path).set_index("smiles")

            expected = ADMET_DATA if include_physchem else ADMET_DATA[ADMET_DATA["category"] != "Physicochemical"]

            assert len(preds) == 1
            assert len(preds.columns) == len(expected)

            for column in preds.columns:
                assert np.allclose(
                    preds[column].values, FIRST_DRUGBANK_ROW[column], rtol=RTOL, atol=ATOL
                ), f"Column {column} does not match"

    @pytest.mark.parametrize("num_workers", [0, 1])
    @pytest.mark.parametrize("include_physchem", [True, False])
    def test_admet_predict_drugbank(self, num_workers: int, include_physchem: bool) -> None:
        """Test predictions on DrugBank data using admet_predict."""
        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            preds_path = Path(temp_dir) / "preds.csv"

            drugbank_smiles = DRUGBANK_DATA[["smiles"]]
            drugbank_smiles.to_csv(data_path, index=False)

            admet_predict(
                data_path=data_path,
                save_path=preds_path,
                include_physchem=include_physchem,
                drugbank_path=None,
                atc_code=None,
                num_workers=num_workers,
            )

            preds = pd.read_csv(preds_path).set_index("smiles")

            expected = ADMET_DATA if include_physchem else ADMET_DATA[ADMET_DATA["category"] != "Physicochemical"]
            assert len(preds.columns) == len(expected)

            for column in preds.columns:
                assert np.allclose(
                    preds[column].values, DRUGBANK_DATA[column].values, rtol=RTOL, atol=ATOL
                ), f"Column {column} does not match"

    @pytest.mark.parametrize("atc_code", [None, "nervous system"])
    def test_admet_predict_drugbank_with_percentiles(self, atc_code: str | None) -> None:
        """Test admet_predict on DrugBank data with DrugBank percentiles."""
        if atc_code is None:
            reference_data = DRUGBANK_DATA_PERCENTILES
        else:
            reference_data = DRUGBANK_DATA_PERCENTILES_NERVOUS_SYSTEM

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.csv"
            preds_path = Path(temp_dir) / "preds.csv"

            drugbank_smiles = reference_data[["smiles"]]
            drugbank_smiles.to_csv(data_path, index=False)

            admet_predict(
                data_path=data_path,
                save_path=preds_path,
                include_physchem=True,
                atc_code=atc_code,
            )

            preds = pd.read_csv(preds_path).set_index("smiles")

            assert len(preds.columns) == 2 * len(ADMET_DATA)

            for column in preds.columns:
                rtol = PERCENTILE_RTOL if _is_percentile_column(column) else RTOL
                atol = PERCENTILE_ATOL if _is_percentile_column(column) else ATOL

                assert np.allclose(
                    preds[column].values, reference_data[column].values, rtol=rtol, atol=atol
                ), f"Column {column} does not match"


class TestADMETModel:
    @pytest.mark.parametrize("include_physchem", [True, False])
    def test_admet_model_single_smiles(self, include_physchem: bool) -> None:
        """Test ADMETModel.predict with the first DrugBank SMILES."""
        model = ADMETModel(
            include_physchem=include_physchem,
            drugbank_path=None,
            atc_code=None,
        )

        preds = model.predict(smiles=FIRST_DRUGBANK_SMILES)

        expected = ADMET_DATA if include_physchem else ADMET_DATA[ADMET_DATA["category"] != "Physicochemical"]

        assert isinstance(preds, dict)
        assert len(preds) == len(expected)

        for key in preds.keys():
            assert np.allclose(
                preds[key], FIRST_DRUGBANK_ROW[key], rtol=RTOL, atol=ATOL
            ), f"{key} prediction does not match"

    @pytest.mark.parametrize("num_workers", [0, 1])
    @pytest.mark.parametrize("include_physchem", [True, False])
    def test_admet_model_drugbank(self, num_workers: int, include_physchem: bool) -> None:
        """Test predictions on DrugBank data using ADMETModel."""
        model = ADMETModel(
            include_physchem=include_physchem,
            drugbank_path=None,
            atc_code=None,
            num_workers=num_workers,
        )

        preds = model.predict(smiles=DRUGBANK_DATA["smiles"].tolist())

        expected = ADMET_DATA if include_physchem else ADMET_DATA[ADMET_DATA["category"] != "Physicochemical"]
        assert len(preds.columns) == len(expected)

        for column in preds.columns:
            assert np.allclose(
                preds[column].values, DRUGBANK_DATA[column].values, rtol=RTOL, atol=ATOL
            ), f"Column {column} does not match"

    @pytest.mark.parametrize("atc_code", [None, "nervous system"])
    def test_admet_model_drugbank_with_percentiles(self, atc_code: str | None) -> None:
        """Test ADMETModel on DrugBank data with DrugBank percentiles."""
        model = ADMETModel(
            include_physchem=True,
            atc_code=atc_code,
        )

        if atc_code is None:
            reference_data = DRUGBANK_DATA_PERCENTILES
        else:
            reference_data = DRUGBANK_DATA_PERCENTILES_NERVOUS_SYSTEM

        preds = model.predict(smiles=reference_data["smiles"].tolist())

        assert len(preds.columns) == 2 * len(ADMET_DATA)

        for column in preds.columns:
            rtol = PERCENTILE_RTOL if _is_percentile_column(column) else RTOL
            atol = PERCENTILE_ATOL if _is_percentile_column(column) else ATOL

            assert np.allclose(
                preds[column].values, reference_data[column].values, rtol=rtol, atol=atol
            ), f"Column {column} does not match"
