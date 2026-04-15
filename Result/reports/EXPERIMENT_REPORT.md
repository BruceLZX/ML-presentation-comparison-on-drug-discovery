# Experiment Report

## Abstract
This project benchmarks machine learning and graph neural network methods for molecular property prediction across 7 datasets covering HTS, toxicity, and ADME tasks. We evaluated classical RDKit-descriptor baselines (logistic regression, random forest, gradient boosting, extra trees, ada boost), Chemprop (MPNN), and Chemprop+RDKit on 3 scaffold-based splits. In addition, we produced SARS-focused interpretability outputs (MCTS rationales), descriptor-level feature analysis (importance + SHAP), and latent-space visualizations (PCA/t-SNE/UMAP). Overall, classical RDKit-feature models were strong and often outperformed Chemprop variants under this small-epoch CPU setting.

## Introduction
The goal is to build a compact but complete experimental pipeline aligned with mentor requirements:
- Dataset focus: SARS-CoV-2 and HIV (HTS), hERG/DILI/AMES (tox), HIA/Bioavailability (ADME).
- Model comparison: baseline ML vs. MPNN (Chemprop), and Chemprop+RDKit.
- Interpretation and visualization:
  - MCTS-style rationale extraction for SARS candidate molecules.
  - Property distribution plots for SARS molecules.
  - RDKit feature selection/attribution.
  - Chemprop latent-space plots.

## Experiments
### Datasets
- `SARSCoV2_3CLPro_Diamond`
- `ogbg-molhiv`
- `hERG`
- `DILI`
- `AMES`
- `HIA_Hou`
- `Bioavailability_Ma`

### Data split and evaluation
- 3 scaffold splits (prepared dataset objects with fixed train/valid/test splits).
- Metric: AUROC (reported as mean ± std over splits).

### Models
- ML baselines (RDKit descriptors): `logistic_regression`, `random_forest`, `gradient_boosting`, `extra_trees`, `ada_boost`.
- Deep learning: `chemprop` (MPNN), `chemprop_rdkit`.

### SARS interpretability and analysis
- MCTS rationale extraction based on Chemprop example logic (adapted to trained SARS model).
- Feature exploration with chemfunc `plot_property_distribution`.
- RDKit feature selection with RandomForest feature importance and SHAP ranking.
- Chemprop latent representation visualization via PCA/t-SNE/UMAP.

## Results
### 1) Baseline and performance table (AUROC %, mean ± std)
From `results/classification_results_all_table.csv`:

| Model | AMES | Bioavailability_Ma | DILI | HIA_Hou | SARSCoV2_3CLPro_Diamond | hERG | ogbg-molhiv |
|---|---:|---:|---:|---:|---:|---:|---:|
| ada_boost | 80.1 ± 4.1 | 68.1 ± 6.6 | 82.9 ± 3.9 | 95.2 ± 5.3 | 70.4 ± 8.8 | 80.1 ± 1.3 | 67.0 ± 12.6 |
| chemprop | 76.8 ± 3.0 | 69.6 ± 8.6 | 80.2 ± 11.4 | 39.5 ± 8.5 | 44.9 ± 7.9 | 58.8 ± 10.9 | 54.8 ± 15.9 |
| chemprop_rdkit | 71.8 ± 2.0 | 38.4 ± 9.4 | 63.9 ± 2.3 | 53.4 ± 5.7 | 36.9 ± 11.7 | 62.6 ± 4.9 | 56.5 ± 9.1 |
| extra_trees | 84.6 ± 2.9 | 74.2 ± 2.8 | 86.1 ± 3.7 | 98.5 ± 1.4 | 90.0 ± 6.4 | 85.0 ± 3.1 | 82.9 ± 0.9 |
| gradient_boosting | 84.0 ± 2.8 | 72.9 ± 6.9 | 81.3 ± 5.7 | 95.7 ± 4.8 | 80.1 ± 11.2 | 79.6 ± 2.4 | 71.0 ± 13.3 |
| logistic_regression | 82.8 ± 1.1 | 74.2 ± 2.4 | 83.1 ± 3.3 | 98.8 ± 1.4 | 74.8 ± 0.9 | 76.6 ± 2.9 | 65.3 ± 18.9 |
| random_forest | 84.8 ± 3.2 | 74.1 ± 0.5 | 84.6 ± 4.6 | 98.3 ± 1.6 | 87.9 ± 7.4 | 84.7 ± 2.9 | 84.3 ± 1.9 |

### 2) MCTS interpretability (SARS)
- Output grid: `plots/mcts/SARSCoV2_3CLPro_Diamond_mcts_rationale_grid.png`
- Detailed CSV: `plots/mcts/SARSCoV2_3CLPro_Diamond_mcts_rationales.csv`
- Example extracted rationales include:
  - `O=[CH:1]C1CCC1` (for two molecules)
  - `c1cnn[cH:1]c1` (for one molecule)

### 3) Feature plots (SARS, chemfunc)
- `plots/property/SARSCoV2_3CLPro_Diamond_MolWt_distribution.pdf`
- `plots/property/SARSCoV2_3CLPro_Diamond_MolLogP_distribution.pdf`
- `plots/property/SARSCoV2_3CLPro_Diamond_TPSA_distribution.pdf`

### 4) RDKit feature selection (SARS)
- Importance CSV: `plots/feature_selection/SARSCoV2_3CLPro_Diamond_rdkit_feature_importance.csv`
- SHAP CSV: `plots/feature_selection/SARSCoV2_3CLPro_Diamond_rdkit_shap_importance.csv`
- Top-feature plots:
  - `plots/feature_selection/SARSCoV2_3CLPro_Diamond_top20_rdkit_feature_importance.png`
  - `plots/feature_selection/SARSCoV2_3CLPro_Diamond_top20_rdkit_shap_importance.png`
- Top SHAP-ranked descriptors include `PEOE_VSA5`, `MolWt`, `ExactMolWt`, `VSA_EState10`, `Chi1v`.

### 5) Chemprop latent space (SARS)
- `plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_pca.png`
- `plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_tsne.png`
- `plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_umap.png`

## Discussion
1. Under current setup (CPU, short training, fixed prepared splits), RDKit-descriptor ML baselines were consistently strong across most datasets, especially `extra_trees` and `random_forest`.
2. Chemprop and Chemprop+RDKit underperformed on several tasks in this run, suggesting sensitivity to training budget, hyperparameters, and data regime.
3. The interpretability and feature-analysis artifacts are now complete and reusable for poster/report creation:
   - Model-level performance table,
   - Molecule-level rationale outputs,
   - Descriptor-level attribution outputs.
4. Optional extension not included in this report: GPCR (Butkiewicz) additional scaffold-split experiment.

## Generated Files (This Bundle)
- `results/classification_results_all_table.csv`
- `results/classification_results_all_long.csv`
- `results/classification_results_focus7_final_splits.csv`
- `plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_pca.png`
- `plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_tsne.png`
- `plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_umap.png`
- `plots/mcts/SARSCoV2_3CLPro_Diamond_mcts_rationale_grid.png`
- `plots/mcts/SARSCoV2_3CLPro_Diamond_mcts_rationales.csv`
- `plots/mcts/SARSCoV2_3CLPro_Diamond_top_predicted_actives.csv`
- `plots/property/SARSCoV2_3CLPro_Diamond_MolWt_distribution.pdf`
- `plots/property/SARSCoV2_3CLPro_Diamond_MolLogP_distribution.pdf`
- `plots/property/SARSCoV2_3CLPro_Diamond_TPSA_distribution.pdf`
- `plots/feature_selection/SARSCoV2_3CLPro_Diamond_rdkit_feature_importance.csv`
- `plots/feature_selection/SARSCoV2_3CLPro_Diamond_rdkit_shap_importance.csv`
- `plots/feature_selection/SARSCoV2_3CLPro_Diamond_top20_rdkit_feature_importance.png`
- `plots/feature_selection/SARSCoV2_3CLPro_Diamond_top20_rdkit_shap_importance.png`
