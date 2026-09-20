# Weekly Report — Week 4
## ML Internship at Darnica

**Project:** MLInternship_week4
**Side Quest:** Evaluate the Classical Classifier
**Main Quest:** Supervised Machine Learning: Regression and Classification / Advanced Learning Algorithms (evaluation & model-development sections)

---

## 1. Executive Summary

This week's side quest focused on building a complete, professional evaluation pipeline for classical machine learning classifiers. Three models — Logistic Regression, Decision Tree, and SVM — were trained on the Breast Cancer Wisconsin dataset and evaluated using accuracy, precision, recall, F1-score, confusion matrices, ROC/AUC, and cross-validation.

**Key result:** Logistic Regression achieved the best overall performance (98.25% test accuracy, AUC 0.9957), with only 2 misclassified samples out of 114 in the test set. Bias/variance analysis confirmed good generalization with no overfitting or underfitting. A statistical significance test (McNemar's) showed that the performance gap between Logistic Regression and SVM is not statistically significant at this sample size — an important nuance beyond raw accuracy numbers.

In parallel, the main quest coursework (Supervised ML: Regression and Classification, Advanced Learning Algorithms — evaluation and model-development sections) was studied; notes are maintained separately in `study-notes/`.

---

## 2. Problem & Dataset

**Task:** Binary classification — distinguish malignant vs. benign breast tumors from diagnostic measurements.

**Dataset:** Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`.

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (all numerical) |
| Classes | 2 — Malignant (0), Benign (1) |
| Class distribution | Benign: 357 (62.74%), Malignant: 212 (37.26%) |
| Missing values | 0 |
| Duplicate rows | 0 |

No cleaning was required beyond feature scaling. The mild class imbalance (ratio ≈ 1.68:1) was noted and later confirmed not to distort per-class performance (see Section 7).

---

## 3. Data Preparation

- **Split strategy:** Stratified train/validation/test split to preserve class proportions.
  - Train: 273 samples (48%)
  - Validation: 182 samples (32%)
  - Test: 114 samples (20%)
- **Scaling:** `StandardScaler` fit on the training set only, then applied to validation and test sets (no data leakage).
- **Reproducibility:** `random_state = 42` fixed across all splits and models.

---

## 4. Models & Training

Three classical classifiers were trained with default hyperparameters as a baseline:

1. Logistic Regression
2. Decision Tree
3. Support Vector Machine (SVM, RBF kernel)

---

## 5. Results

### 5.1 Validation Set

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9835 | 0.9826 | 0.9912 | 0.9869 |
| Decision Tree | 0.9231 | 0.9464 | 0.9298 | 0.9381 |
| SVM | 0.9670 | 0.9737 | 0.9737 | 0.9737 |

### 5.2 Test Set

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9957** |
| Decision Tree | 0.9123 | 0.9429 | 0.9167 | 0.9296 | 0.9107 |
| SVM | 0.9737 | 0.9726 | 0.9861 | 0.9793 | 0.9937 |

Validation and test scores are close for all models, indicating stable, non-overfit estimates.

### 5.3 Confusion Matrix — Logistic Regression (Test Set)

|  | Predicted Malignant | Predicted Benign |
|---|---|---|
| **Actual Malignant** | 41 (TN) | 1 (FP) |
| **Actual Benign** | 1 (FN) | 71 (TP) |

Only 2 errors out of 114 test samples — 1 false positive and 1 false negative.

---

## 6. Failure / Error Analysis

Two misclassified samples were identified for Logistic Regression on the test set:

1. **False Negative** — True label: Benign, Predicted: Malignant
2. **False Positive** — True label: Malignant, Predicted: Benign

Given the very small number of errors, no systematic error pattern (e.g., a consistently confused class) was detected — the two mistakes appear to be isolated borderline cases rather than a structural weakness of the model.

### Feature Importance (Logistic Regression, |coefficient|)

Top contributing features, in order:

1. worst texture
2. radius error
3. worst radius
4. worst area
5. worst symmetry
6. worst concave points
7. mean concave points
8. worst perimeter
9. area error
10. worst concavity

The dominance of "worst"-prefixed features (representing the most extreme cell measurements per sample) suggests the model relies heavily on the severity of the most abnormal cells rather than average measurements — consistent with clinical intuition for tumor malignancy.

---

## 7. Bias / Variance Analysis

| Set | Accuracy |
|---|---|
| Train | 0.9853 |
| Validation | 0.9835 |
| Test | 0.9825 |

- Train–Validation gap: 0.0018
- Train–Test gap: 0.0029
- **Diagnosis: Good generalization — no overfitting, no underfitting.**

**5-Fold Cross-Validation:** Mean accuracy 0.9633 ± 0.0165 (single-fold variation observed, but overall consistent).

**Learning curves:** Training and validation scores converge as training set size increases, with no persistent gap — confirming the bias/variance diagnosis above.

**Class imbalance check:** Despite a 62.7%/37.3% class split, per-class F1-scores were nearly identical (Malignant: 0.9762, Benign: 0.9861 — difference of 0.0099), confirming the model is not biased toward the majority class. No resampling or class-weighting was necessary.

---

## 8. Extra Exploration (Beyond Week 4 Requirements)

*The following was done out of personal curiosity and interest in learning more — not part of the core assigned deliverables.*

### 8.1 Hyperparameter Tuning (GridSearchCV)

| Model | Best Parameters | Best CV F1 | Test F1 (before → after) |
|---|---|---|---|
| Decision Tree | `max_depth=3, min_samples_leaf=1, min_samples_split=2` | 0.9414 | 0.9296 → 0.9362 |
| SVM | `C=10, gamma='scale', kernel='rbf'` | 0.9742 | 0.9793 → 0.9504 |

**Observation:** Tuning improved the Decision Tree as expected, but the tuned SVM performed *worse* on the test set than the default configuration, despite a higher cross-validation score. This illustrates an important lesson: the hyperparameters that maximize CV performance are not guaranteed to generalize best to an unseen test set, especially with a small dataset where fold-to-fold variance can mislead model selection.

### 8.2 Precision-Recall Curves

| Model | Average Precision (AP) |
|---|---|
| Logistic Regression | 0.9973 |
| SVM | 0.9961 |
| Decision Tree | 0.9169 |

Precision-Recall curves are arguably more informative than ROC for this medical diagnosis context, since a false negative (missed malignant case) carries far higher real-world cost than a false positive.

### 8.3 Statistical Significance Testing (McNemar's Test)

Comparing Logistic Regression vs. SVM on the test set:

|  | SVM Correct | SVM Wrong |
|---|---|---|
| **LR Correct** | 111 | 1 |
| **LR Wrong** | 0 | 2 |

- McNemar's statistic: 0.0
- p-value: 1.0000
- **Result: Not statistically significant (p ≥ 0.05)**

Although Logistic Regression scored numerically higher (98.25% vs. 97.37% accuracy), this test shows the difference could plausibly be due to chance at this sample size. This is a meaningful distinction between "numerically better" and "statistically significantly better."

---

## 9. Conclusions & Recommendations

- **Best model:** Logistic Regression, based on the highest test accuracy, F1-score, and AUC, and the fewest test-set errors.
- **Generalization:** All diagnostics (train/val/test gap, cross-validation, learning curves) confirm the model generalizes well with no over/underfitting.
- **Statistical caveat:** The advantage of Logistic Regression over SVM is not statistically significant on this test set — with more data, this conclusion could change.
- **Class imbalance:** Present but mild, and does not measurably bias per-class performance; no correction was necessary.
- **Possible future improvements:**
  - Evaluate on a larger or external dataset to confirm the McNemar's test result with more statistical power.
  - Explore ensemble methods (e.g., Random Forest, Gradient Boosting) as a natural extension of the Decision Tree baseline.
  - Apply SHAP values for a more rigorous feature-importance analysis than raw logistic regression coefficients.

---

## Artifacts

All supporting figures are saved in `reports/figures/`:
- `eda_overview.png`
- `confusion_matrix_lr.png`
- `roc_curves.png`
- `model_comparison.png`
- `feature_importance.png`
- `cross_validation.png`
- `learning_curves.png`
- `precision_recall_curves.png`

Full implementation: `notebooks/classifier_evaluation.ipynb`
