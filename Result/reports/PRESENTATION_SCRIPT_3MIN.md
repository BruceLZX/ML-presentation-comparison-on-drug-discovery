# Three-Minute Presentation Script

## Title
Machine Learning for SARS-CoV-2 Antiviral Drug Discovery

## Main Script
Hello, my name is Bruce. My project studies machine learning for molecular property prediction, with a focus on SARS-CoV-2 antiviral screening.

The motivation is that drug discovery is slow, costly, and experimentally intensive. Virtual screening can help prioritize molecules before wet-lab testing. In addition, high-throughput screening datasets are often skewed or imbalanced, which makes model evaluation more difficult. At the same time, graph neural networks are popular in molecular machine learning, but strong classical descriptor-based baselines are often under-emphasized. That led to two research questions.

First, how well does Chemprop, a message passing neural network, perform on skewed datasets that are common in high-throughput screening? Second, are the resulting predictions interpretable?

To study this, I used benchmark data from the Therapeutics Data Commons, or TDC. I evaluated seven binary classification datasets spanning three areas: high-throughput screening, toxicity, and ADMET. These included SARS-CoV-2 3CLPro and HIV for screening, hERG, DILI, and AMES for toxicity, and HIA and Bioavailability for ADMET.

Each dataset was evaluated using three scaffold-based splits. A scaffold split separates molecules by core chemical scaffold, so it is a stricter test of chemical generalization than a random split. I used AUROC as the main metric because these are binary classification tasks and AUROC is robust when class balance varies.

I compared seven models in total. The classical RDKit-descriptor baselines were logistic regression, random forest, gradient boosting, extremely randomized trees, and AdaBoost. I also evaluated Chemprop, which is a message passing neural network where atoms are nodes and bonds are edges, and a Chemprop plus RDKit descriptor variant.

The main result was that descriptor-based tree models performed best under the current benchmark setup. Extremely Randomized Trees achieved the best average AUROC across all datasets at 85.9, and Random Forest was close behind at 85.5. In comparison, Chemprop achieved 60.7 on average, and Chemprop plus RDKit achieved 54.8.

The clearest case study was the SARS-CoV-2 3CLPro dataset. On that task, Extremely Randomized Trees reached 90.0 plus or minus 6.4, and Random Forest reached 87.9 plus or minus 7.4. The current Chemprop runs were substantially lower, at 44.9 plus or minus 7.9 for Chemprop and 36.9 plus or minus 11.7 for Chemprop plus RDKit.

For interpretability, the most useful results were descriptor-level. I used RDKit feature importance, SHAP analysis, and molecular property distribution plots on the SARS dataset. The top descriptors included features such as molecular weight, surface-area-related descriptors, and electronic-property descriptors. The LogP distribution plot also showed partial separation between active and inactive compounds, which supports descriptor-based screening behavior.

One important limitation is that the Chemprop experiments here should be interpreted as baseline deep-learning runs, not fully optimized graph neural network benchmarks. In the saved configuration, the Chemprop runs used a short training budget, so these results should not be taken as the best possible Chemprop performance.

Overall, this project shows that strong classical descriptor-based baselines remain essential in molecular screening, especially on skewed high-throughput screening tasks. It also shows that descriptor-level interpretation can provide useful chemical insight, even when model-level interpretability remains limited. Thank you.

## Short Q&A Prep
### What is Chemprop?
- Chemprop is a message passing neural network for molecules.
- It treats atoms as nodes and bonds as edges, then learns a graph representation for prediction.

### What is a scaffold split?
- A scaffold split groups molecules by core scaffold, so train and test molecules are chemically more distinct than in a random split.

### Why use AUROC?
- AUROC is threshold-independent and works well for binary classification, especially when classes are imbalanced.

### What is Extra Trees?
- Extra Trees means Extremely Randomized Trees, a tree-ensemble model related to Random Forest but with more randomized split selection.

### Did you preprocess the data yourself?
- The safest answer is that benchmark data provided through TDC was used.

### Why did Chemprop underperform here?
- The most grounded answer is that these were baseline Chemprop runs under a modest training setup, while the RDKit baselines were strong and well matched to these datasets.

### Were the predictions interpretable?
- Yes, at the descriptor level through SHAP, feature importance, and property-distribution analysis.
- The MCTS rationale experiment was exploratory and weaker.
