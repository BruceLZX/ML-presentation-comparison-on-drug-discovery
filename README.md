# ML Presentation Comparison on Drug Discovery

This repository contains my project code, experiments, figures, and presentation materials for benchmarking molecular property prediction models on drug discovery tasks, with a focus on SARS-CoV-2 antiviral screening.

The project compares classical RDKit-descriptor baselines against Chemprop under a benchmark setup built on TDC datasets.

## Repository Purpose

This repository is no longer being used as the original upstream `admet_ai` project README.
Instead, it serves as my project workspace and submission repository for:

- benchmark experiments on HTS, toxicity, and ADMET datasets
- Chemprop and Chemprop + RDKit comparisons
- SARS-focused case study analysis
- poster, presentation, and video materials
- final result tables, figures, and reports

## Main Project Question

The project was guided by two questions:

1. How does Chemprop perform on skewed datasets that are common in high-throughput screening?
2. Are the resulting predictions interpretable?

## Main Findings

- Tree-based RDKit-descriptor baselines performed best under the current benchmark setup.
- Extremely Randomized Trees achieved the best average AUROC across the selected datasets: `85.9`.
- Random Forest was second overall: `85.5`.
- The clearest case study was the SARS dataset, where descriptor-based models clearly outperformed the current Chemprop baseline runs.
- Descriptor-level analysis using SHAP and property distributions provided supporting interpretation for the SARS case study.

## Repository Structure

- `scripts/`
  - experiment and analysis scripts
  - baseline ML training/evaluation
  - Chemprop prepared-split experiments
  - SHAP / feature analysis
  - property-distribution plotting
  - latent-space plotting
- `Result/`
  - final result bundle used for poster and presentation preparation
- `Result/results_csv/`
  - benchmark result CSV files
- `Result/plots/`
  - final plots for case study and reporting
- `Result/reports/`
  - project details
  - experiment report
  - poster content
  - presentation scripts
  - video scripts
- `REPRODUCTION.md`
  - notes for the earlier Docker-based reproduction workflow
- `Dockerfile.repro`
  - Docker environment for reproduction work

## Important Files

- `Result/reports/PROJECT_DETAILS.md`
  - detailed project notes, definitions, Q&A prep, and presentation guidance
- `Result/reports/PRESENTATION_SCRIPT_3MIN.md`
  - 3-minute presentation script
- `Result/reports/VIDEO_WALKTHROUGH_SCRIPT_2MIN.md`
  - 2-minute video script
- `Result/reports/POSTER_CONTENT.md`
  - poster text and figure planning
- `Result/results_csv/classification_results_all_table.csv`
  - main model comparison table
- `Result/results_csv/classification_results_focus7_final_splits.csv`
  - split-level results, including SARS case study values

## Core Models Compared

Classical RDKit-descriptor baselines:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extremely Randomized Trees
- AdaBoost

Deep learning models:

- Chemprop
- Chemprop + RDKit descriptors

## Data Source

- Benchmark data from the Therapeutics Data Commons (TDC)
- Prepared benchmark dataset objects provided for fixed scaffold-split evaluation

## Reproduction Notes

If you want to review the earlier Docker-based subset reproduction work, see:

- `REPRODUCTION.md`
- `Dockerfile.repro`
- `scripts/run_subset_repro_in_docker.sh`

## Git Remote

This repository is intended to be tracked at:

`git@github.com:BruceLZX/ML-presentation-comparison-on-drug-discovery.git`

## Clone

```bash
git clone git@github.com:BruceLZX/ML-presentation-comparison-on-drug-discovery.git
cd ML-presentation-comparison-on-drug-discovery
```
