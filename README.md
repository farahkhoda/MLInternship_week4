# MLInternship_week4

**Evaluate the Classical Classifier** — Week 4 Internship Project at Darnica

---

## Project Summary

This project completes two parallel tracks of the Week 4 ML internship:

**Main Quest:** Supervised Machine Learning (Regression & Classification) + Advanced Learning Algorithms — focused on evaluation methodology and model-development strategies.

**Side Quest:** Build and evaluate a classical binary classifier on real medical data (Breast Cancer Wisconsin dataset) using proper ML engineering practices.

---

## What I Learned (Main Quest)

### Supervised Learning: Regression & Classification
- **Linear Regression:** Cost function (MSE), gradient descent, feature scaling, learning rate selection
- **Logistic Regression:** Sigmoid function, decision boundaries, binary cross-entropy loss, why it's better than linear regression for classification
- **Regularization:** L1 (Lasso) and L2 (Ridge) techniques to prevent overfitting and reduce variance
- **Feature Engineering:** Polynomial features, normalization, selection methods to improve model performance
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and AUC-ROC for comprehensive model assessment

**Key Insight:** Different models and metrics are appropriate for different problems; choosing correctly requires understanding the task and domain.

### Advanced Learning Algorithms: Evaluation & Model Development
- **Train/Validation/Test Split:** Why three sets are essential; preventing test set contamination; proper methodology for fair model comparison
- **Bias/Variance Tradeoff:** High bias (underfitting) vs. high variance (overfitting); diagnosing which problem a model has; how to fix each
- **Learning Curves:** Visualizing training vs. validation performance to identify whether more data or model complexity is needed
- **Error Analysis:** Manually inspecting misclassified examples to identify systematic patterns and guide improvements
- **Model Selection:** Using validation set to choose between models before reporting results on test set
- **Classical Algorithms:** Decision Trees (entropy, information gain, stopping criteria), Support Vector Machines (margin, kernel trick), K-Nearest Neighbors, Ensemble methods (bagging, boosting)

**Key Insight:** Evaluation methodology is as important as the algorithm; a rigorous evaluation pipeline catches overfitting and ensures honest performance estimates.

---

## What I Built (Side Quest)

### The Project: Binary Classification for Medical Diagnosis

**Task:** Classify breast tumors as malignant or benign using 30 numerical features from diagnostic imaging.

**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features, 2 classes — 62.7% benign, 37.3% malignant)

**Approach:**
1. **Exploratory Data Analysis:** Verified data quality (no missing values, no duplicates), analyzed feature distributions, identified feature scaling needs
2. **Data Preparation:** Applied stratified train/validation/test split (48%/32%/20%) and StandardScaler normalization
3. **Model Training:** Trained three classifiers — Logistic Regression, Decision Tree, SVM — and compared performance
4. **Comprehensive Evaluation:** Computed accuracy, precision, recall, F1-score, confusion matrices, ROC/AUC curves, cross-validation scores, and learning curves
5. **Error Analysis:** Identified 2 misclassifications out of 114 test samples; no systematic pattern detected
6. **Bias/Variance Diagnosis:** Confirmed excellent generalization (training-test gap of 0.29%), no overfitting or underfitting

### Results

**Best Model:** Logistic Regression
- Test Accuracy: **98.25%**
- Test Precision: 98.61% | Recall: 98.61% | F1: 98.61%
- AUC-ROC: **0.9957**
- Misclassifications: Only 2 out of 114 test samples

**Why Logistic Regression won:**
- Simpler model; lower overfitting risk
- Interpretable coefficients (feature importance)
- Fast inference
- Good generalization with minimal tuning

---

## Extra Exploration (Beyond Requirements)

Went beyond the core checklist to deepen understanding:

### 1. Hyperparameter Tuning (GridSearchCV)
- Systematically searched Decision Tree and SVM hyperparameter spaces
- Decision Tree F1 improved: 0.9296 → 0.9362
- SVM test performance varied: illustrates important lesson that best CV scores ≠ best test scores
- **Lesson:** Cross-validation optimizes for the validation set, not the test set; variance between folds can mislead

### 2. Precision-Recall Curves
- Computed for all three models alongside ROC curves
- Precision-Recall is more informative for medical diagnosis (false negatives are catastrophic)
- Logistic Regression AP: 0.9973 (best)

### 3. Statistical Significance Testing (McNemar's Test)
- Compared Logistic Regression vs. SVM performance statistically
- Result: p-value = 1.0 (not significant)
- **Lesson:** Numerical accuracy difference (98.25% vs 97.37%) doesn't mean one model is truly better; sample size matters

---

## Key Insights

1. **Proper evaluation is non-negotiable:** The train/validation/test methodology prevented over-optimistic performance claims and ensured honest assessment.

2. **Bias/variance diagnosis guides fixes:** Instead of blindly adding more data or regularization, diagnose the problem first. Small training-test gap here meant no overfitting — no need for intervention.

3. **Metrics matter more than a single number:** Accuracy alone (98.25%) doesn't tell the full story. Precision, recall, F1, confusion matrix, ROC/AUC, and error analysis together create a complete picture.

4. **Simplicity often wins:** Logistic Regression beat more complex models (Decision Tree, SVM). Occam's Razor: prefer simpler models that generalize better.

5. **Statistical rigor:** McNemar's test showed that without statistical testing, I might have claimed Logistic Regression was clearly better when it was actually just random variation.

6. **Feature understanding:** The most important features (worst texture, radius error, worst radius) align with medical intuition — "worst" extreme values matter more for malignancy than averages.

---

## Deliverables

- **Notebook:** `classifier_evaluation.ipynb` — complete pipeline with all analysis steps, visualizations, and code
- **Weekly Report:** `weekly_report_week4.md` — detailed methodology, results, analysis, and conclusions
- **Study Notes:** 
  - `regression_classification_notes.md` — comprehensive notes on linear/logistic regression and regularization
  - `advanced_learning_algorithms_notes.md` — detailed notes on evaluation methodology, bias/variance, and classical algorithms
- **Code Modules:** `src/preprocessing.py`, `src/train.py`, `src/evaluate.py` — reusable, clean, well-documented functions
- **Visualizations:** 8+ plots covering EDA, confusion matrices, ROC curves, feature importance, learning curves, model comparisons

---

## Skills Demonstrated

✅ End-to-end ML pipeline design
✅ Proper data splitting and validation methodology
✅ Multiple model training and comparison
✅ Comprehensive evaluation (metrics, visualizations, statistical tests)
✅ Error analysis and bias/variance diagnosis
✅ Code organization (modular, documented, reproducible)
✅ Professional documentation and reporting
✅ Going beyond minimum requirements (hyperparameter tuning, statistical testing)
