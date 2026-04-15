# Poster Content

## Poster Title
Machine Learning for SARS-CoV-2 Antiviral Drug Discovery

## Optional Title Refinements
- Machine Learning Benchmarking for SARS-CoV-2 Antiviral Virtual Screening
- Benchmarking Classical and Deep Models for SARS-CoV-2 Antiviral Screening

## Authors
Zhexiang (Bruce) Li, Emily Nguyen, Yan Liu

## Affiliation
Department of Computer Science, University of Southern California

## Poster Layout
- Left column: Motivation, Methods
- Center top: Main Results
- Center bottom: SARS Case Study
- Right column: Limitations & Future Directions, Conclusion

## Motivation
- Drug discovery is costly and time-intensive.
- Virtual screening can prioritize molecules before wet-lab testing.
- SARS-CoV-2 antiviral screening is a useful case study for rapid molecular ranking.
- Strong baseline comparisons matter because descriptor-based models are often under-reported.

## Motivation Figure
- Make one simple conceptual figure manually in Google Slides:
  - Large chemical space
  - Virtual screening narrows candidates
  - Experimental validation on fewer compounds

## Methods
- Benchmarked 7 binary classification datasets spanning HTS, toxicity, and ADMET.
- Used 3 scaffold-based splits for each dataset.
- Reported AUROC mean +/- std across splits.
- Compared 7 models:
  - Logistic regression
  - Random forest
  - Gradient boosting
  - Extra trees
  - AdaBoost
  - Chemprop
  - Chemprop + RDKit
- Performed SARS-focused feature and property analysis.
- Guiding question:
  - Can RDKit-descriptor baselines remain competitive across molecular classification tasks, and does SARS-CoV-2 3CLPro provide the clearest test case?

## Methods Figure
- Make one workflow diagram manually:
  - Dataset -> scaffold split -> model training -> AUROC evaluation -> SARS analysis

## Methods Table
| Component | Choice |
|---|---|
| Datasets | 7 classification datasets |
| Domains | HTS, toxicity, ADMET |
| Splits | 3 scaffold-based splits |
| Metric | AUROC |
| Classical ML | LR, RF, GB, ET, AdaBoost |
| Deep learning | Chemprop, Chemprop+RDKit |

## Dataset Summary Table
Keep this small. This is supporting context, not a main visual.

| Dataset | Domain | Molecules |
|---|---|---|
| SARSCoV2_3CLPro_Diamond | HTS | 879 |
| ogbg-molhiv | HTS | 40983 |
| hERG | Toxicity | 622 |
| DILI | Toxicity | 475 |
| AMES | Toxicity | 7255 |
| HIA_Hou | ADMET | 578 |
| Bioavailability_Ma | ADMET | 640 |

## Main Results
- Tree-based descriptor models consistently outperformed the current Chemprop runs across benchmark domains.
- Performance gains were not limited to one task type, but extended across HTS, toxicity, and ADMET datasets.
- Extra Trees achieved the best average AUROC across all datasets: 85.9.
- Random Forest was a close second: 85.5.
- Classical baselines remained highly competitive for practical molecular ranking.

## Results Highlight Box
- Best overall model: Extra Trees, average AUROC 85.9
- Second-best overall model: Random Forest, average AUROC 85.5
- Best SARS result: Extra Trees, 90.0 +/- 6.4

## Main Performance Table
- Build manually from:
  - `final_submission_all_in_one/results_csv/classification_results_all_table_presentable.md`
- Keep:
  - Best result bold
  - Second-best result underlined
  - Average column included

## Optional Ranking Table
Use only if you clearly have extra space.

| Model | Average AUROC |
|---|---|
| extra_trees | 85.9 |
| random_forest | 85.5 |
| gradient_boosting | 80.6 |
| logistic_regression | 79.4 |
| ada_boost | 77.7 |
| chemprop | 60.7 |
| chemprop_rdkit | 54.8 |

## SARS Case Study
- SARS-CoV-2 3CLPro produced the clearest separation between strong classical baselines and the deep-learning baselines tested here.
- Extra Trees and Random Forest delivered both high AUROC and consistent ranking signal across scaffold splits.
- Descriptor-level interpretation and property-shift analysis further support chemically meaningful screening behavior.

## SARS-Only Split Table
| Model | Split 1 | Split 2 | Split 3 | Mean +/- Std |
| extra_trees | 99.0 | 85.5 | 85.5 | 90.0 +/- 6.4 |
| random_forest | 98.5 | 82.8 | 82.6 | 87.9 +/- 7.4 |
| gradient_boosting | 95.9 | 72.9 | 71.4 | 80.1 +/- 11.2 |
| logistic_regression | 75.5 | 73.4 | 75.3 | 74.8 +/- 0.9 |
| ada_boost | 82.7 | 62.4 | 66.0 | 70.4 +/- 8.8 |
| chemprop | 33.7 | 50.5 | 50.5 | 44.9 +/- 7.9 |
| chemprop_rdkit | 20.4 | 45.1 | 45.2 | 36.9 +/- 11.7 |

## SARS Case Study Text
- SARS provides the strongest evidence chain in this project.
- The strongest models were Extra Trees and Random Forest.
- This supports RDKit-descriptor baselines for practical SARS screening.

## SARS Figures To Use
- SHAP feature plot:
  - `final_submission_all_in_one/plots/feature_selection/SARSCoV2_3CLPro_Diamond_top20_rdkit_shap_importance.png`
- Property distribution plot:
  - `final_submission_all_in_one/plots/property/SARSCoV2_3CLPro_Diamond_MolLogP_distribution.png`

## Figure Captions
- SHAP plot:
  - Top SARS descriptors are dominated by size, surface-area, and electronic-property terms.
- LogP plot:
  - Active and inactive SARS compounds occupy partly different physicochemical regions, supporting descriptor-based screening.

## SARS Descriptor Summary
- Do not use a full descriptor table unless you have extra space.
- Prefer a small text box next to the SHAP plot:
  - Top descriptors include PEOE_VSA5, MolWt, ExactMolWt, and VSA_EState10.
  - The strongest signals are related to molecular size, surface area, and electronic properties.

## Limitations & Future Directions
- Chemprop was trained under a modest setup with short training and limited tuning.
- Current deep-learning results should be interpreted as baseline Chemprop results, not fully optimized GNN performance.
- Feature interpretation should be rerun directly on the best-performing SARS model.
- Future work should improve SARS-focused Chemprop training and revisit model-aligned interpretation.

## Conclusion
- Classical RDKit-descriptor baselines were strong across the benchmark.
- Extra Trees and Random Forest were the top-performing models overall.
- For SARS-CoV-2 3CLPro, classical ML clearly outperformed the current Chemprop variants.
- Strong descriptor-based baselines remain essential in molecular screening, and SARS-CoV-2 3CLPro provides a clear HTS test case.

## Final Figure Recommendation
- Use these as the main visuals:
  - Main performance table
  - SARS-only split table
  - SHAP feature plot
  - LogP distribution plot
- Keep methods and dataset tables visually small.
- Add the average-ranking table only if you have space.

## Do Not Use As Main Panels
- `final_submission_all_in_one/plots/mcts/SARSCoV2_3CLPro_Diamond_mcts_rationale_grid.png`
- `final_submission_all_in_one/plots/interpretability/SARSCoV2_3CLPro_Diamond_top_predicted_panel.png`
- `final_submission_all_in_one/plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_tsne.png`
- `final_submission_all_in_one/plots/latent/SARSCoV2_3CLPro_Diamond_chemprop_umap.png`

## Footer Text
- This work was conducted at the University of Southern California.
- Thanks to Emily Nguyen and Yan Liu for project guidance and feedback.
