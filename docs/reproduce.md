# Reproducing ADMET-AI Models

This document provides step-by-step instructions for reproducing the ADMET-AI models in the current release. To reproduce the data, models, and results from [our paper](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030), please check out a commit from ADMET-AI v1 (e.g., v1.4.0 [here](https://github.com/swansonk14/admet_ai/releases/tag/v_1.4.0)) and follow those instructions.

- [Download TDC ADMET data](#download-tdc-admet-data)
- [Create multitask datasets for regression and classification](#create-multitask-datasets-for-regression-and-classification)
- [Create a single dataset with all TDC ADMET data](#create-a-single-dataset-with-all-tdc-admet-data)
- [Train Chemprop ADMET predictors](#train-Chemprop-admet-predictors)
- [Evaluate Chemprop ADMET predictors](#evaluate-chemprop-admet-predictors)
- [Get approved drugs from DrugBank](#get-approved-drugs-from-drugbank)
- [Make predictions on DrugBank approved drugs](#make-predictions-on-drugbank-approved-drugs)

## Download TDC ADMET data

For the following download command only, use a Python environment with Python version 3.10 that contains `PyTDC==1.1.15` and `typed-argument-parser==1.11.0`. (Note that PyTDC is incompatible with the other requirements in ADMET-AI.)

Download all TDC [ADME](https://tdcommons.ai/single_pred_tasks/adme/) and [Tox](https://tdcommons.ai/single_pred_tasks/tox/) datasets for training models. Skip datasets that are redundant or not needed.

```bash
python scripts/prepare_tdc_admet_all.py \
    --save_dir data/tdc_admet_all \
    --skip_datasets herg_central hERG_Karim ToxCast
```

## Filter for valid molecules

Filter all the CSV files to ensure SMILES result in valid RDKit molecules. (Make sure to use the general ADMET-AI Python environment with an up-to-date RDKit.)

```bash
python scripts/filter_valid_molecules.py \
    --data_dir data/tdc_admet_all \
    --smiles_column smiles
```

## Create multitask datasets for regression and classification

Create multitask datasets for regression and classification for all the TDC ADMET datasets.

```bash
python scripts/merge_tdc_admet_multitask.py \
    --data_dir data/tdc_admet_all \
    --save_dir data/tdc_admet_all_multitask
```

## Train Chemprop ADMET predictors

Train Chemprop ADMET predictors on the TDC ADMET multitask datasets. Note: A GPU is used by default if available.

```bash
python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all_multitask \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop
```

## Evaluate Chemprop ADMET predictors

Evaluate Chemprop ADMET predictors trained on the TDC ADMET Benchmark Group data. (Note: The PyTDC environment should be used for this command.)

```bash
python scripts/evaluate_tdc_admet_all_multitask.py \
    --data_dir data/tdc_admet_all_multitask \
    --preds_dir models/tdc_admet_all_multitask/chemprop \
    --save_dir results/tdc_admet_all_multitask
```

## Get approved drugs from DrugBank

Get approved drugs from [DrugBank](https://go.drugbank.com/) (v5.1.14) to create a comparison set for Chemprop ADMET predictors. This assumes that the file `drugbank.xml` has been downloaded from DrugBank (license required).

```bash
python scripts/get_drugbank_approved.py \
    --data_path data/drugbank/drugbank.xml \
    --save_path data/drugbank/drugbank_approved.csv
```

## Make predictions on DrugBank approved drugs

Make ADMET predictions on DrugBank approved drugs using Chemprop multitask predictor (and compute physicochemical properties).

```bash
admet_predict \
    --data_path data/drugbank/drugbank_approved.csv \
    --save_path admet_ai/resources/data/drugbank_approved.csv \
    --models_dir models/tdc_admet_all_multitask/chemprop \
    --smiles_column smiles \
    --drugbank_path None
```
