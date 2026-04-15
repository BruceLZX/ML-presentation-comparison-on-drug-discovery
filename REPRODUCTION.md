# ADMET-AI Reproduction (Chemprop v2, Docker)

This repository was reproduced in Docker with Chemprop v2 using two TDC ADMET group datasets:
- `caco2_wang` (regression)
- `hia_hou` (classification)

## 1) Run end-to-end in Docker

```bash
bash scripts/run_subset_repro_in_docker.sh
```

This runs:
1. `scripts/prepare_tdc_admet_subset.py`
2. `scripts/train_tdc_admet_group.py`
3. `scripts/evaluate_tdc_admet_group.py`

## 2) Output locations

- Prepared data: `repro_data/prepared`
- Trained models: `repro_results/models/chemprop`
- Per-seed predictions: `repro_results/models/chemprop/*/*/model_0/test_predictions.csv`

## 3) Observed evaluation results

From `scripts/evaluate_tdc_admet_group.py`:

Single models across 5 folds:
- `caco2_wang`: `0.337 +- 0.021` (MAE)
- `hia_hou`: `0.929 +- 0.009` (ROC-AUC)

Ensemble across 5 folds:
- `caco2_wang`: `mae = 0.334`
- `hia_hou`: `roc-auc = 0.933`

## Notes

- Docker image file: `Dockerfile.repro`
- This uses ADMET-AI v2 (Chemprop v2), as requested.
