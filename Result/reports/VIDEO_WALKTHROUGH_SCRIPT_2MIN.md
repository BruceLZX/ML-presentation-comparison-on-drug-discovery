Hello, my name is Bruce. My project studies machine learning for molecular property prediction, with a focus on SARS-CoV-2 antiviral screening.

This project asked two questions: how well Chemprop performs on skewed high-throughput screening data, and whether the predictions are interpretable.

To study this, I benchmarked seven binary classification datasets from the Therapeutics Data Commons across high-throughput screening, toxicity, and ADMET tasks. I used three scaffold-based splits and evaluated performance with AUROC.

I compared Chemprop with classical RDKit-descriptor baselines and also tested a Chemprop plus RDKit variant.

The main finding was that descriptor-based tree models performed best in this benchmark. Across all datasets, Extremely Randomized Trees achieved the best average AUROC at 85.9, with Random Forest close behind at 85.5. On the SARS-CoV-2 3CLPro dataset, Extremely Randomized Trees reached 90.0 plus or minus 6.4, while the current Chemprop runs were lower.

For interpretability, I focused on the SARS dataset using SHAP analysis, feature importance, and molecular property distribution plots. These results showed that important signals were linked to molecular size, surface area, and electronic properties.

The main takeaway is that strong classical baselines remain essential for skewed molecular screening data. One limitation is that Chemprop was only evaluated in a baseline setting. Future work will improve Chemprop training on skewed HTS tasks and strengthen model-aligned interpretability.

Overall, this project highlights the importance of both strong baseline comparison and interpretable analysis in AI for drug discovery. Thank you.