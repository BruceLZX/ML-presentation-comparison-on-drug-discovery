# Project Details

## Category
Computational Biology / AI for Drug Discovery

## Title
Machine Learning for SARS-CoV-2 Antiviral Drug Discovery

## Abstract
Drug discovery is slow, costly, and experimentally intensive, so virtual screening is useful for prioritizing molecules before wet-lab testing. This project evaluates molecular property prediction models on seven binary classification benchmark datasets spanning high-throughput screening, toxicity, and ADMET, with particular emphasis on the TDC dataset `SARSCoV2_3CLPro_Diamond`. The models compared are five descriptor-based baselines, `logistic_regression`, `random_forest`, `gradient_boosting`, `extra_trees`, and `ada_boost`, together with `chemprop` and `chemprop_rdkit`. Performance is reported as AUROC mean +/- standard deviation over three scaffold-based splits. In addition to benchmark evaluation, the project includes descriptor-level interpretation through RDKit feature importance, SHAP ranking, and molecular property distribution plots. Under the current benchmark setup, tree-based RDKit descriptor models were the strongest performers overall, while Chemprop should be interpreted as a baseline deep-learning run rather than a fully optimized graph neural network benchmark.

## Description of Individual Contributions
1. Reproducible experimental pipeline
- Implemented and ran an end-to-end workflow for loading benchmark datasets, evaluating fixed scaffold-based splits, training models, and aggregating results.
- Standardized outputs into split-level, long-format, and table-format CSV files.

2. Baseline model framework and modularity
- Built and extended a modular sklearn baseline framework with interchangeable model backends.
- Evaluated five RDKit-descriptor baseline classifiers: logistic regression, random forest, gradient boosting, extremely randomized trees, and AdaBoost.

3. Chemprop and Chemprop + RDKit experiments
- Implemented prepared-split Chemprop training and evaluation scripts.
- Added a Chemprop + RDKit descriptor input mode and generated comparative results against non-RDKit Chemprop and the classical ML baselines.

4. Interpretability and analysis
- Implemented RDKit feature-importance and SHAP analysis for the SARS dataset.
- Generated SARS molecular property distribution plots for molecular weight, LogP, and TPSA.
- Ran an exploratory MCTS-style rationale extraction experiment based on Chemprop example logic.

5. Reporting assets
- Consolidated poster-ready tables, figures, scripts, and reports into a single submission bundle.

## Project Overview
### Research Questions
1. How does Chemprop, a molecular property prediction message passing neural network, perform on skewed datasets that are common in high-throughput screening?
2. Are the resulting predictions interpretable?

### Why This Project Matters
- High-throughput screening datasets are often class-imbalanced, which makes practical model evaluation harder.
- Graph neural networks are popular for molecular learning, but strong classical descriptor baselines are often under-emphasized.
- A fair comparison matters because in applied screening settings, a simpler model that ranks molecules well can be more useful than a more complex model that is not yet well tuned.

## Terminology And Definitions
### What is Chemprop?
- Chemprop is a message passing neural network, or MPNN.
- Molecules are represented as graphs:
  - atoms are nodes
  - bonds are edges
- During message passing, each atom updates its representation using information from neighboring atoms and bonds.
- The final graph representation is then used to predict a molecular property.

### What are RDKit descriptors?
- RDKit descriptors are hand-crafted numerical molecular features computed from a molecule's SMILES string using RDKit.
- Examples include:
  - molecular weight
  - LogP
  - topological polar surface area
  - counts or shape-related descriptors
- In this environment, the descriptor set had 217 columns.

### What is a scaffold split?
- A scaffold split separates molecules by core chemical scaffold rather than random row assignment.
- This is a stricter evaluation setting because it tests whether a model can generalize to chemically different structures, not just near-duplicates.
- In this project, each dataset was evaluated on 3 scaffold-based splits.

### What is AUROC?
- AUROC stands for Area Under the Receiver Operating Characteristic curve.
- It measures how well a classifier ranks positives above negatives across classification thresholds.
- It is useful here because these are binary classification tasks and several datasets are class-imbalanced.

### What is SHAP?
- SHAP stands for SHapley Additive exPlanations.
- In this project, SHAP was used on a RandomForest-based descriptor model for the SARS dataset.
- The plotted value is mean absolute SHAP value, which measures how strongly a descriptor contributes to predictions on average.

### What is MCTS in this project?
- MCTS stands for Monte Carlo Tree Search.
- Here it refers to an exploratory rationale-extraction procedure adapted from the Chemprop interpretability example.
- The goal was to identify small molecular substructures that preserve a model's prediction.
- In this project, that analysis was weaker than the descriptor-level analysis and should be treated as exploratory only.

## Datasets
### Dataset Scope
- High-throughput screening:
  - `SARSCoV2_3CLPro_Diamond`
  - `ogbg-molhiv`
- Toxicity:
  - `hERG`
  - `DILI`
  - `AMES`
- ADMET:
  - `HIA_Hou`
  - `Bioavailability_Ma`

### Data Source
- Benchmark data from TDC was used.
- For the prepared-split experiments, mentor-provided benchmark dataset objects were loaded from joblib files.
- The safest presentation wording is:
  - "We used benchmark data provided through TDC."

## Experimental Setup
### Task Type
- Binary classification

### Split Strategy
- 3 scaffold-based splits per dataset
- For the prepared dataset objects, each split contains train, validation, and test indices

### Metric
- AUROC mean +/- standard deviation over splits

### Random Seed
- Seed 42 for the baseline ML experiments

### Feature Types
- Baseline ML:
  - RDKit descriptors
- Chemprop:
  - learned graph representations from molecular graphs
- Chemprop + RDKit:
  - graph input plus RDKit descriptor matrix

### Actual Chemprop Training Setup Used In These Runs
- The saved result files show that the Chemprop runs in this benchmark used:
  - `epochs = 3`
  - `patience = 2`
- This is one reason Chemprop should be described as a baseline run rather than a fully optimized benchmark.

## Models Evaluated
### Classical ML Models
- `logistic_regression` = Logistic Regression
- `random_forest` = Random Forest
- `gradient_boosting` = Gradient Boosting
- `extra_trees` = Extremely Randomized Trees
- `ada_boost` = AdaBoost

### Deep Learning Models
- `chemprop` = Chemprop (MPNN)
- `chemprop_rdkit` = Chemprop + RDKit descriptors

## Implementation Details Grounded In The Code
### Baseline ML Framework
From `admet_ai/scripts/tdc_single_scaffold_cv_baseline.py`:
- Logistic regression:
  - median imputation
  - standard scaling
  - logistic regression with `max_iter=3000`
- Random forest:
  - median imputation
  - `n_estimators=500`
- Gradient boosting:
  - median imputation
  - sklearn `GradientBoostingClassifier`
- Extremely randomized trees:
  - median imputation
  - sklearn `ExtraTreesClassifier` with `n_estimators=500`
- AdaBoost:
  - median imputation
  - sklearn `AdaBoostClassifier`

### Chemprop Pipeline
From `admet_ai/scripts/run_chemprop_prepared_classification.py`:
- Prepared dataset objects were loaded from joblib files.
- SMILES and target columns were auto-detected from the dataset object.
- Invalid SMILES were filtered using Chemprop's own molecule parser logic.
- For `chemprop_rdkit`, RDKit descriptor matrices were computed from RDKit descriptor functions and saved as `.npz`.
- AUROC on each test split was computed from Chemprop prediction outputs.

## Main Results
### Overall Results Table Summary
- Best average model overall:
  - Extremely Randomized Trees, `85.9`
- Second-best overall:
  - Random Forest, `85.5`
- Third overall:
  - Gradient Boosting, `80.6`
- Chemprop:
  - `60.7`
- Chemprop + RDKit:
  - `54.8`

### Dataset-Level Best Results
- `AMES`:
  - Random Forest, `84.8 +/- 3.2`
- `Bioavailability_Ma`:
  - Extremely Randomized Trees, `74.2 +/- 2.8`
  - effectively tied with Logistic Regression, `74.2 +/- 2.4`
- `DILI`:
  - Extremely Randomized Trees, `86.1 +/- 3.7`
- `HIA_Hou`:
  - Logistic Regression, `98.8 +/- 1.4`
- `SARSCoV2_3CLPro_Diamond`:
  - Extremely Randomized Trees, `90.0 +/- 6.4`
- `hERG`:
  - Extremely Randomized Trees, `85.0 +/- 3.1`
- `ogbg-molhiv`:
  - Random Forest, `84.3 +/- 1.9`

### SARS Split-Level Results
From `classification_results_focus7_final_splits.csv`:
- Extremely Randomized Trees:
  - split 1: `98.98`
  - split 2: `85.45`
  - split 3: `85.45`
  - mean: `90.0 +/- 6.4`
- Random Forest:
  - split 1: `98.47`
  - split 2: `82.80`
  - split 3: `82.58`
  - mean: `87.9 +/- 7.4`
- Chemprop:
  - split 1: `33.67`
  - split 2: `50.52`
  - split 3: `50.52`
  - mean: `44.9 +/- 7.9`
- Chemprop + RDKit:
  - split 1: `20.41`
  - split 2: `45.12`
  - split 3: `45.17`
  - mean: `36.9 +/- 11.7`

## Interpretation And Analysis
### Descriptor-Level Interpretation
From `SARSCoV2_3CLPro_Diamond_rdkit_shap_importance.csv`, top SHAP-ranked descriptors were:
- `PEOE_VSA5`
- `MolWt`
- `ExactMolWt`
- `VSA_EState10`
- `Chi1v`
- `fr_alkyl_halide`
- `SMR_VSA10`
- `LabuteASA`

Interpretation:
- `MolWt` and `ExactMolWt` relate to molecular size
- `LabuteASA` and `VSA_EState10` relate to surface-area or volume-related structure
- `PEOE_VSA5` and related descriptors reflect electronic-property patterns distributed over molecular surface bins

### Property Distribution Analysis
From `plot_sars_property_distributions.py`:
- Three RDKit properties were plotted:
  - molecular weight
  - LogP
  - TPSA
- The x-axis was clipped to the 1st to 99th percentile to avoid a few outliers dominating the plot.
- The strongest presentation-friendly plot was LogP because it showed visible partial separation between active and inactive classes.

### Chemprop Latent-Space Analysis
- PCA, t-SNE, and UMAP visualizations were generated from Chemprop fingerprints.
- These plots are supplementary analysis outputs, not the strongest evidence in the final story.

### MCTS Rationale Analysis
From `SARSCoV2_3CLPro_Diamond_mcts_rationales.csv`:
- Only 4 top-predicted molecules were exported in that run.
- Their predicted scores were all low, around `0.332` to `0.335`.
- All 4 had `target = 0`, meaning they were inactive examples in the benchmark labels.
- `rationale_smiles` and `rationale_score` were empty for these examples.

Interpretation:
- The MCTS rationale output was weak in this project.
- It should not be presented as a strong positive interpretability result.
- The stronger interpretability result was the descriptor-level analysis.

## What Can Be Claimed Safely
### Safe Claims
- Under the current benchmark setup, tree-based RDKit models outperformed the current Chemprop runs on most datasets.
- Descriptor-level interpretation on SARS was informative.
- Strong baseline comparisons matter in molecular screening.
- Chemprop was included as a valid baseline deep-learning comparison.

### Claims To Avoid
- Do not say that Chemprop was fully optimized.
- Do not say that Chemprop itself was strongly interpretable from the MCTS results.
- Do not imply heavy custom preprocessing if benchmark data from TDC was used.
- Do not say that tree-based models are always better than graph neural networks in general.

## Main Scientific Takeaways
1. Strong classical descriptor-based baselines remain essential in molecular screening.
2. In this benchmark, tree-based RDKit models were the strongest overall performers.
3. The SARS dataset provided the clearest case study for both benchmark comparison and descriptor-level interpretation.
4. The current Chemprop results should be interpreted as baseline deep-learning results under a modest training budget.

## Presentation Notes
### How To Refer To The SARS Dataset
- For readability on slides or posters, it is acceptable to say:
  - `SARS`
  - `SARS-CoV-2 3CLPro`
- If needed, clarify once that this refers to the TDC benchmark dataset `SARSCoV2_3CLPro_Diamond`.

### How To Explain Data Processing
- Best wording:
  - "We used benchmark data provided through TDC."
- If asked for more detail:
  - "The prepared dataset objects already contained benchmark splits, and we trained and evaluated models on those benchmark splits."

### How To Explain Extra Trees
- Extremely Randomized Trees is a tree-ensemble model related to Random Forest.
- The difference is that split selection is made more random, which can reduce variance and sometimes improve generalization.

### How To Explain Chemprop Simply
- Chemprop is a message passing neural network for molecular graphs.
- Atoms are treated as nodes, bonds are treated as edges, and the model learns a graph representation used for prediction.

### How To Explain Poster Figures
#### SARS Split Table
- This table compares all models on the SARS dataset across 3 scaffold-based splits.
- The main point is consistency:
  - Extremely Randomized Trees performed best overall
  - Random Forest was second
  - the current Chemprop runs were substantially lower
- Safe explanation:
  - "This table shows that the tree-based descriptor models were consistently strongest on SARS across scaffold splits."

#### SARS Average-AUROC Bar Chart
- This bar chart is a visual summary of the SARS model comparison.
- It does not add a new metric; it makes the ranking easier to see quickly.
- Safe explanation:
  - "This bar chart summarizes the same SARS comparison visually and makes the ranking easier to see at a glance."

#### SHAP Plot
- This plot is not from Chemprop.
- It was generated from a RandomForest-based RDKit descriptor model on the SARS dataset.
- The x-axis is mean absolute SHAP value:
  - longer bars mean a descriptor had a larger average influence on predictions
- The top features are mainly related to:
  - molecular size
  - surface area
  - electronic properties
- Safe explanation:
  - "This SHAP plot shows which RDKit descriptors mattered most for the SARS Random Forest model. The most important features are mainly related to molecular size, surface area, and electronic properties."
- Important caution:
  - because this is mean absolute SHAP, it captures importance magnitude, not direction of effect

#### LogP Distribution Plot
- This plot is dataset-level, not model-specific.
- It compares the LogP distribution of active and inactive SARS molecules.
- LogP reflects molecular lipophilicity, which is one relevant physicochemical property.
- The distributions are not completely separated, but they show partial shift.
- Safe explanation:
  - "This plot shows that active and inactive SARS molecules have partially shifted LogP distributions, which suggests that physicochemical properties carry useful screening signal."
- Important caution:
  - do not say that LogP alone determines activity

#### How The Four Main SARS Figures Fit Together
- Split table and bar chart:
  - performance comparison
- SHAP plot and LogP distribution:
  - supporting chemical interpretation

## Potential Q&A
### Why did classical ML outperform Chemprop here?
- The most grounded answer is that Chemprop was run under a modest setup, while the RDKit descriptor baselines were strong and matched well to these datasets.
- The saved run configuration indicates Chemprop used only 3 epochs with patience 2, so it should not be framed as a fully tuned GNN benchmark.

### Did you preprocess the datasets yourself?
- The safe answer is no custom preprocessing should be emphasized.
- Say:
  - "We used benchmark data provided through TDC."

### Why use AUROC instead of accuracy?
- AUROC is threshold-independent and is more informative when class balance varies.
- That matters here because HTS datasets can be skewed.

### Are the predictions interpretable?
- Yes, at the descriptor level.
- SHAP, feature importance, and property-distribution analysis were informative.
- The MCTS rationale output was exploratory and weaker.

### Was Chemprop a fair comparison?
- Yes, as a baseline comparison.
- No, if someone interprets it as a fully optimized deep-learning benchmark.
- The correct framing is:
  - valid baseline
  - not final optimized GNN performance

### Why focus on SARS?
- SARS was one of the mentor-prioritized high-throughput screening tasks.
- It also produced the clearest separation between strong descriptor-based baselines and the current Chemprop runs.
- It had the most interpretable supporting analysis in the final project story.

### What should be improved next?
- Longer and better tuned Chemprop training on skewed HTS datasets
- Better alignment between interpretation and the best-performing model
- Additional experiments on related screening datasets

## Presentation Script Aligned To The Poster
### Three-Minute Oral Script
Hello, my name is Bruce. My project studies machine learning for molecular property prediction, with a focus on SARS-CoV-2 antiviral screening.

This project was guided by two questions: how well Chemprop performs on skewed high-throughput screening data, and whether the predictions are interpretable.

To study this, I benchmarked seven binary classification datasets from the Therapeutics Data Commons across high-throughput screening, toxicity, and ADMET tasks. I used three scaffold-based splits and evaluated performance with AUROC.

I compared Chemprop with classical RDKit-descriptor baselines and also tested a Chemprop plus RDKit variant. Chemprop is a message passing neural network, or MPNN, which represents molecules as graphs where atoms are nodes and bonds are edges.

The main finding was that descriptor-based tree models performed best in this benchmark. Across all datasets, Extremely Randomized Trees achieved the best average AUROC at 85.9, with Random Forest close behind at 85.5.

The clearest case study was SARS. On the SARS-CoV-2 3CLPro dataset, Extremely Randomized Trees reached 90.0 plus or minus 6.4, while the current Chemprop runs were lower. This made SARS the clearest example of the gap between strong descriptor-based baselines and the current Chemprop baseline.

The SARS section also includes supporting descriptor-level analysis through SHAP and molecular property distribution plots. These are supporting evidence rather than the main claim.

One limitation is that Chemprop was only evaluated in a baseline setting with short training and limited tuning, so these results should not be interpreted as fully optimized graph neural network performance.

Overall, this project shows that strong classical descriptor-based baselines remain essential for skewed molecular screening data, and under the current benchmark setup they outperformed the current Chemprop runs, especially on the SARS case study. Thank you.

### What To Emphasize During The Talk
- Main focus:
  - motivation
  - benchmark setup
  - model comparison
  - main performance result
  - SARS as the clearest case study
- Secondary support only:
  - SHAP plot
  - property-distribution plot
- Do not over-emphasize:
  - latent-space plots
  - MCTS rationale analysis

### What Not To Spend Time On In The Talk
- Do not turn interpretability into the main story.
- Do not present MCTS as a strong positive result.
- Do not spend time on implementation details such as `.npz` caching, parser filtering, or script structure unless directly asked.

### Poster-Aligned Takeaway
- The poster's main message is:
  - strong RDKit-descriptor baselines outperformed the current Chemprop runs
  - SARS was the clearest case study
  - descriptor-level analysis provided supporting evidence, not the central claim
