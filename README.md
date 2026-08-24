# EEG Seizure Prediction — Generalization Study

A systematic study of how preprocessing pipeline design, regularization strategy, and class-imbalance handling interact to affect generalization in EEG-based seizure detection, evaluated across three datasets with very different seizure prevalence rates.

## 🎯 Project Overview

Seizure prediction from EEG is a high-stakes classification problem: datasets vary enormously in class imbalance (from ~1% to ~22% seizure rate), and a model that generalizes well on one distribution may fail on another. Rather than optimizing a single model on a single dataset, this project asks a broader question: **which combinations of preprocessing, regularization, and imbalance-handling generalize best, and why?**

The study runs a full factorial comparison across:
- **3 datasets** with different seizure prevalence and noise characteristics
- **2 preprocessing pipelines** (reduction-first vs. extraction-first)
- **3 regularization strategies** (L1, L2, ElasticNet)
- **2 imbalance-handling techniques** (SMOTE, class weighting)

## 📊 Datasets

| Dataset | Samples | Features | Seizure Rate | Notes |
|---|---|---|---|---|
| **CHB-MIT Scalp EEG** | 100,000 | 178 | ~5% | 23-patient corpus; severe imbalance |
| **UCI Epileptic Seizure Recognition** | 11,500 | 178 | ~21.7% | Time-series features collapsed to binary labels |
| **TUH EEG Seizure Corpus** | 50,000 | 178 | ~1% | Extreme imbalance, high-noise scenario |

The notebook attempts to fetch each dataset via the Kaggle/PhysioNet APIs, and **falls back to statistically-matched simulated data** (matching class imbalance ratios, feature dimensions, and noise profiles) when API credentials or institutional data-use agreements aren't available — so the methodology runs end-to-end even without real data access. Instructions for obtaining the real datasets (Kaggle, PhysioNet, TUH) are included in an appendix section.

## 🧪 Methodology

**1. Preprocessing pipelines (Section 2)** — two competing philosophies are compared head-to-head:
- **Pipeline A (reduction-first):** StandardScaler → Gaussian temporal smoothing (σ=1.5) → SelectKBest (ANOVA F-test, top 50 features)
- **Pipeline B (extraction-first):** MinMaxScaler → PCA (50 components) → FastICA (20 independent components)

**2. Baseline modeling (Section 3)** — Logistic Regression baseline across all dataset × pipeline combinations, evaluated on accuracy, macro-F1, and PR-AUC.

**3. Overfitting/underfitting demonstration (Section 4)** — three deliberately-constructed scenarios (underfit with 5 features + heavy regularization, an optimal fit, and an overfit case with polynomial features + injected noise), compared via train/test F1 gap and learning curves.

**4. Regularization study (Section 5)** — L1, L2, and ElasticNet penalties compared across every dataset/pipeline combination, measuring F1, PR-AUC, coefficient sparsity, and stability across 5 random seeds.

**5. Class imbalance handling (Section 6)** — on the hardest dataset (TUH, 1% seizure rate), compares:
- **SMOTE oversampling + L2** regularization
- **Class weighting + L1** regularization

evaluated via precision-recall curves and precision at a fixed recall of 0.80.

**6. Comparative analysis (Section 7)** — all results are consolidated into a summary table and heatmaps (F1 and PR-AUC by configuration × regularization type), with explicit answers to four generalization questions: pipeline design effects, best-generalizing regularizer, when ElasticNet fails to win, and how imbalance-handling interacts with regularization choice.

## 📈 Key Findings

- **Preprocessing order matters.** Pipeline A (denoise-first) tends to outperform Pipeline B (extract-first) in most settings, since noisy features are filtered out before dimensionality reduction rather than being blended into the reduced components.
- **ElasticNet generalizes best on average**, combining L1 sparsity with L2 smoothness — though its `l1_ratio` requires careful tuning.
- **ElasticNet doesn't always win.** On the high-noise TUH dataset, L2 alone can outperform ElasticNet by avoiding over-sparsification of the rare seizure pattern; after ICA decorrelation (Pipeline B), L1 can also beat ElasticNet since the correlated-feature advantage ElasticNet relies on no longer applies.
- **Imbalance handling creates a precision-recall tradeoff.** SMOTE + L2 increases seizure recall but raises false positives (shifting the decision boundary into non-seizure territory). Class Weighting + L1 is more conservative, penalizing minority-class errors without altering the training distribution — yielding fewer false alarms. This makes **Class Weighting + L1 preferable for clinical deployment** (avoiding alarm fatigue), while **SMOTE + ElasticNet suits research settings** aiming to maximize detection.

## 📁 Repository Structure

```
.
├── EEG_Seizure_Prediction.ipynb   # Full pipeline: data loading → preprocessing → modeling → imbalance handling → analysis
├── pr_curves_imbalance.png        # Precision-recall curves (generated on run)
├── summary_heatmap.png            # F1 / PR-AUC heatmaps (generated on run)
└── README.md
```

*(Rename the notebook from its default/uploaded name to something descriptive, e.g. `EEG_Seizure_Prediction_Generalization_Study.ipynb`.)*

## 🚀 Running the Notebook

```
pip install scikit-learn imbalanced-learn pandas matplotlib seaborn scipy numpy
jupyter notebook EEG_Seizure_Prediction.ipynb
```

To use real data instead of the simulated fallback:
- **UCI Epileptic Seizure (Kaggle):** save a Kaggle API key to `~/.kaggle/kaggle.json`, then `kaggle datasets download -d harunshimanto/epileptic-seizure-recognition`
- **CHB-MIT (PhysioNet):** free registration required; load with `mne.io.read_raw_edf(...)`
- **TUH EEG Seizure Corpus:** requires an institutional data-use agreement — use the built-in simulated data for methodology development in the meantime

## ⚠️ Notes & Limitations

- Real dataset access requires Kaggle credentials, PhysioNet registration, or a TUH data-use agreement respectively; without these, the notebook automatically substitutes simulated data matched to each dataset's real-world class balance, feature dimensionality, and noise characteristics.
- Findings on which regularizer or imbalance technique "wins" are dataset- and pipeline-dependent — the comparative tables should be read as demonstrating *interaction effects*, not a single universally-best configuration.

## 🎓 Context

Developed as part of coursework/research within a Data Science MS program (IMSciences, Peshawar), focused on model generalization, regularization, and imbalanced classification in a biomedical signal processing context.

## 📄 License

Educational/academic use.
