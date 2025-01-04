# Fluorescent protein classification using Machine Learning

This repository contains the source files and supplementary information for a machine learning pipeline designed to classify protein structures based on sequence-derived features. The project integrates preprocessing, encoding strategies, classification models, optimization, and results visualization.

---

## Table of Contents

- [Summary of Proposed Work](#summary)
- [Requirements and Installation](#requirements)
- [Dataset and Preprocessing](#dataset)
- [Feature Encoding](#encoding)
- [Classification Models and Optimization](#models)
- [Performance Analysis](#performance)

---

<a name="summary"></a>
## Summary of proposed work

The project focuses on classifying protein by leveraging sequence-derived features and machine learning methods. The pipeline includes:

- Data preprocessing and descriptive analysis.
- Sequence encoding using physicochemical properties, one-hot encoding, and embeddings (e.g., ESM1b).
- Training multiple classification models.
- Model optimization using Optuna for hyperparameter tuning.
- Comprehensive performance evaluation and visualization.

Key algorithms include:
- Classification: SVC, RandomForest, GradientBoosting, AdaBoost, and more.
- Encoding methods: One-hot, FFT, and embeddings like Bepler and ESM1b.

---

<a name="requirements"></a>
## Requirements and installation

- Python 3.9+
- pandas
- scikit-learn
- optuna
- matplotlib
- seaborn
- joblib

---

<a name="dataset"></a>
## Dataset and preprocessing

The datasets are located in the `input` directory:
- `osfp-full-data-set.csv`: Main dataset.
- `aaindex_encoders.csv`: Index of physicochemical properties.

Key preprocessing steps are implemented in the following notebooks:
- Dataset description: `01_dataset_description_propertys.ipynb`, `01_dataset_description_secondary_structure.ipynb`
- Preprocessing: `01_dataset_preprocessing.ipynb`

Processed datasets and outputs are stored in `results/characterizing_dataset`.

---

<a name="encoding"></a>
## Feature encoding

Sequence features are encoded using various strategies:
- One-hot encoding
- Physicochemical properties (e.g., ANDN920101, ARGP820101, CRAJ730102)
- Embeddings (e.g., Bepler, ESM1b)

Key notebook:
- `02_encoding_peptide_sequences.ipynb`

Encoded datasets are saved in `results/encoders`.

---

<a name="models"></a>
## Classification models and optimization

The pipeline trains and optimizes multiple classification models for protein classification:
- Algorithms: SVC, RandomForest, GradientBoosting, AdaBoost, ExtraTrees, DecisionTree, KNN
- Optimization: Optuna for hyperparameter tuning.

Key notebooks:
- Model training: `04_training_class_model.ipynb`
- Optimization: `05_optimization_optuna.py`

Results are saved in `results/performance`.

---

<a name="performance"></a>
## Performance analysis

Performance evaluation includes:
- Accuracy, precision, recall, and F1-score.
- Model-specific metrics and comparison by encoding method.

Key visualizations:
- Algorithm performance: `ml_classic_performance_by_algorithm.png`
- Encoder performance: `ml_classic_performance_by_encoder.png`

Results are located in `results/performance` and `results/summary_exploring`.

