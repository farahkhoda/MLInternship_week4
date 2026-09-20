# Advanced Learning Algorithms
## Study Notes — Week 4
### Scope: Evaluation & Model-Development sections only (Neural Network sections excluded)

---

## 1. Evaluating a Model

### Train/Test Split
Splitting data into training and test sets is the fundamental step for evaluating how well a model generalizes to unseen data.

**Typical split ratios:**
- 80% train / 20% test (small-to-medium datasets)
- 70% train / 30% test

**Key rules:**
- Always shuffle before splitting (unless time-series data)
- Use stratified split for classification to preserve class proportions
- Fix `random_state` for reproducibility

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Training Error vs. Test Error

**Training error (J_train):** Error measured on the training set. Always optimistically low; model has already seen this data.

**Test error (J_test):** Error measured on the held-out test set. True measure of generalization.

**Key insight:**
- If J_train is low but J_test is high → Overfitting
- If both J_train and J_test are high → Underfitting
- If J_train ≈ J_test and both are low → Good generalization

### Why a Separate Test Set
If you tune model hyperparameters on the test set repeatedly, the test set effectively becomes part of training — optimistic estimates, bad generalization reports.

**Solution:** Introduce a third validation (dev) set. See Section 2.

---

## 2. Model Selection: Train/Validation/Test Sets

### Why Three Sets Instead of Two
When comparing multiple models or tuning hyperparameters, using only train/test leads to **test set contamination** — the test set gets implicitly optimized for, producing over-optimistic results.

**Solution:** Use three splits:
- **Train:** Fit model parameters
- **Validation (Dev/Cross-Validation):** Select best model/hyperparameters
- **Test:** Final, unbiased evaluation — used only ONCE

### Training Set Purpose
Learn the model's parameters (weights/coefficients) by minimizing the cost function on this data.

### Validation Set Purpose
- Select between multiple model architectures
- Tune hyperparameters (e.g., learning rate, regularization λ, tree depth)
- Monitor for overfitting during training
- **Never used for final performance reporting**

### Test Set Purpose
- Report final model performance
- Simulates how the model performs on completely unseen real-world data
- Must remain untouched until all decisions are finalized
- **Never use test set for any tuning or selection**

### Choosing Between Models Using the Validation Set

1. Train all candidate models on the training set
2. Evaluate each on the validation set
3. Select the model with the best validation performance
4. Report final performance on the test set (only once)

```
Model A → val_acc = 95.2%  ←  Winner
Model B → val_acc = 93.1%
Model C → val_acc = 91.8%

→ Report: Model A test_acc = 94.8%
```

---

## 3. Bias and Variance

### High Bias (Underfitting)
**Symptoms:**
- High training error (J_train is large)
- High test/validation error (J_test ≈ J_train, both high)
- Model too simple to capture patterns in data

**Causes:**
- Model too simple (e.g., linear model for non-linear data)
- Too few features
- Too much regularization (λ too large)

**Solutions:**
- Use a more complex model
- Add more features or polynomial features
- Reduce regularization (decrease λ)
- Train longer / use better optimization

### High Variance (Overfitting)
**Symptoms:**
- Low training error (J_train is small)
- High test/validation error (J_test >> J_train)
- Large gap between training and test performance

**Causes:**
- Model too complex relative to data size
- Too many features
- Insufficient training data
- Too little regularization

**Solutions:**
- Collect more training data
- Reduce model complexity
- Feature selection or dimensionality reduction
- Increase regularization (increase λ)
- Use dropout (for neural networks)
- Early stopping

### Bias/Variance Tradeoff
As model complexity increases:
- Bias decreases (model can capture more patterns)
- Variance increases (model becomes more sensitive to training data)

**Goal:** Find the sweet spot of complexity that minimizes both bias and variance, resulting in lowest test error.

```
Error
  |
  |  \
  |   \    Variance
  |    \  /
  |     \/
  |     /\
  |    /  \
  |   / Bias\
  |__/__________
        Complexity
```

### Learning Curves
Plots of training/validation error vs. number of training examples.

**What they show:**
- How performance changes as more data is added
- Whether more data would help
- Whether bias or variance is the dominant problem

**High Bias pattern:**
- Both train and validation error are high
- Curves converge at a high error value
- Adding more data: won't help much (model is too simple)

**High Variance pattern:**
- Training error is low, validation error is high
- Large gap between the two curves
- Adding more data: likely helps (gap will close)

---

## 4. Regularization and Bias/Variance

### L1 Regularization (Lasso)
**Cost function addition:** + (λ/2m) × Σ|θⱼ|

**Effect:**
- Drives some coefficients exactly to zero
- Natural feature selection
- Results in sparse models
- Useful when many features are irrelevant

### L2 Regularization (Ridge)
**Cost function addition:** + (λ/2m) × Σθⱼ²

**Effect:**
- Shrinks all coefficients toward zero (but not exactly zero)
- More stable solution than Lasso when features are correlated
- Most commonly used regularization method

### Effect of Lambda (λ) on Bias/Variance

| Lambda Value | Effect | Result |
|---|---|---|
| λ = 0 | No regularization | High variance risk |
| λ very large | All θⱼ → 0 | High bias (underfitting) |
| λ optimal | Balanced shrinkage | Good generalization |

### Choosing the Regularization Parameter
1. Try a range of λ values: 0, 0.01, 0.1, 1, 10, 100
2. Train model for each λ on training set
3. Evaluate each on validation set
4. Select λ with lowest validation error
5. Report final performance on test set

---

## 5. Establishing a Baseline Level of Performance

### Human-Level Performance
Before interpreting bias/variance, establish a **baseline** — what performance level is achievable at all?

**Common baselines:**
- Human-level performance (for perception tasks)
- Performance of a simple heuristic or previous system
- Published results on the same dataset

### Comparing Model Performance to Baseline
The interpretation of "high" bias or variance depends on the baseline:

**Example:**
- Baseline (human-level): 95% accuracy
- Training accuracy: 94% → High bias relative to baseline
- Validation accuracy: 93% → Small variance issue

vs.

- Baseline: 80% accuracy
- Training accuracy: 94% → No bias problem
- Validation accuracy: 85% → High variance

### Identifying Whether to Focus on Bias or Variance
| Gap | Problem | Action |
|---|---|---|
| J_train >> Baseline | High Bias | More complex model, more features |
| J_val >> J_train | High Variance | More data, regularization, simpler model |
| Both gaps large | Both | Address bias first, then variance |
| Both gaps small | Good | Consider deployment |

---

## 6. Error Analysis

### Manually Inspecting Misclassified Examples
**Process:**
1. Run model on validation set
2. Collect all misclassified examples
3. Manually examine 20-100 of them
4. Look for patterns

**Why this matters:** Automated metrics (accuracy, F1) tell you *how much* the model is wrong; error analysis tells you *why*.

### Categorizing Common Error Types
Create a table of error categories and count:

| Error Type | Count | % of Errors | Notes |
|---|---|---|---|
| Borderline cases | 5 | 50% | Features of both classes |
| Label noise | 2 | 20% | Possibly mislabeled |
| Rare class | 3 | 30% | Too few training examples |

This helps prioritize where to focus improvement efforts.

### Practical Application in This Week's Project
In the Breast Cancer classifier:
- Total test errors: 2 out of 114 samples
- Error 1: True Benign predicted as Malignant (False Positive)
- Error 2: True Malignant predicted as Benign (False Negative)
- Too few errors to identify a clear pattern → model is performing well

---

## 7. Adding Data

### When to Collect More Data
**Collect more data if:**
- High variance diagnosed (learning curves show gap between train/val)
- More data is feasible and affordable

**Don't collect more data if:**
- High bias — more data won't help a model that's too simple
- Data is already large and variance is low

### Data Augmentation
Create modified copies of existing data to artificially expand the dataset.

**Examples:**
- Images: rotate, flip, crop, change brightness
- Audio: add noise, shift pitch, change speed
- Text: synonym replacement, back-translation

**Rule:** Augmentation must produce examples that are representative of real-world test data. Do not create distortions that would never appear in practice.

### Transfer Learning
Use a model pre-trained on a large dataset, then fine-tune on your smaller task-specific dataset.

**Why it works:** Lower layers learn general features (edges, textures, patterns); upper layers learn task-specific features. Fine-tuning reuses general knowledge.

**Steps:**
1. Download pre-trained model
2. Replace final layer(s) with task-specific output layer
3. Fine-tune on your data (optionally freeze early layers)

---

## 8. Model Development Strategies

### Iterative Improvement Loop
ML development is an iterative process:

```
1. Start with a simple baseline model
2. Train and evaluate
3. Diagnose: bias problem or variance problem?
4. Apply appropriate fix
5. Repeat until performance is satisfactory
```

**Common mistake:** Spending weeks tuning before establishing a baseline. Start simple, measure, then improve.

### Debugging ML Systems
When performance is poor, systematically check:
- Is training data sufficient and clean?
- Is the optimization working (is training loss decreasing)?
- Is the model architecture appropriate for the task?
- Are features correctly preprocessed (scaling, encoding)?
- Is there data leakage (test data seen during training)?
- Are evaluation metrics appropriate for the task?

### Prioritizing What to Work On
Use error analysis + bias/variance diagnosis to prioritize:
- High bias → don't waste time collecting more data
- High variance → don't waste time adding model complexity
- Focus on the fix that matches the diagnosed problem
- Use the validation set to measure if the fix helped

---

## 9. Classical Algorithms

### Decision Trees

**How they work:**
1. At each node, choose the feature and threshold that best splits the data
2. Recursively split until stopping criteria are met
3. Leaf nodes contain the majority class prediction

**Entropy and Information Gain:**

Entropy: H(S) = -Σ p_i × log₂(p_i)

Information Gain: IG = H(parent) - weighted_average(H(children))

**Choose the split that maximizes Information Gain.**

**When to stop splitting:**
- Maximum depth reached
- Minimum samples per leaf reached
- No further information gain
- All samples in a node belong to one class

**Advantages:**
- Interpretable (can visualize the tree)
- No feature scaling needed
- Handles both numerical and categorical features
- Fast inference

**Disadvantages:**
- Prone to overfitting (especially deep trees)
- Unstable — small changes in data can change the tree significantly
- Biased toward features with many values

### Support Vector Machines (SVM)

**Maximal Margin Principle:**
Find the hyperplane that maximizes the margin between the two classes.

**Margin** = distance from hyperplane to nearest point in each class (support vectors).

Larger margin → better generalization.

**Kernel Trick:**
Maps data to a higher-dimensional space where it becomes linearly separable, without explicitly computing the transformation.

Common kernels:
- **Linear:** K(x, z) = xᵀz — for linearly separable data
- **RBF (Gaussian):** K(x, z) = exp(-γ||x-z||²) — most widely used, handles non-linear boundaries
- **Polynomial:** K(x, z) = (xᵀz + c)^d

**Hyperparameter C:**
- Small C: Allows misclassifications; larger margin (more regularization)
- Large C: Penalizes misclassifications heavily; smaller margin (less regularization)

**When to use SVM:**
- Small-to-medium datasets
- High-dimensional data (text classification)
- When you need a clear margin of separation
- When training data is limited

### K-Nearest Neighbors (KNN)

**How it works:**
1. Store all training examples
2. For a new point, find the K nearest training points (by Euclidean distance)
3. Predict majority class among K neighbors

**Choosing K:**
- Small K (e.g., K=1): High variance, sensitive to noise
- Large K: High bias, smoother decision boundary
- Common: Use odd K to avoid ties; try K = √n where n = training samples

**Important:** Feature scaling is crucial for KNN — features with larger ranges dominate the distance calculation.

### Ensemble Methods

**Bagging (Bootstrap Aggregating):**
- Train multiple models on different random subsets of training data (with replacement)
- Aggregate predictions by majority vote (classification) or average (regression)
- Reduces variance
- **Example:** Random Forest = bagging of decision trees + random feature subsets

**Boosting:**
- Train models sequentially, each one focusing on examples previous model got wrong
- Combine models by weighted voting
- Reduces bias and variance
- **Examples:** AdaBoost, Gradient Boosting, XGBoost

**Random Forests:**
- Build many decision trees using bagging
- Each tree uses a random subset of features at each split
- Predictions = majority vote across all trees
- More robust than single decision tree
- Feature importance can be extracted from average impurity decrease

---

## 10. Connection to This Week's Side Quest: Breast Cancer Classifier

### Train/Val/Test Splitting Applied
Used a 3-way stratified split:
- Train: 273 samples (48%)
- Validation: 182 samples (32%)
- Test: 114 samples (20%)

**Stratified** → class proportions preserved (62.7% Benign / 37.3% Malignant) in all three sets.

Validation set was used to compare models (LR vs DT vs SVM). Test set used only once for final reporting.

### Bias/Variance Diagnosis Applied
| Set | Accuracy |
|---|---|
| Train | 98.53% |
| Validation | 98.35% |
| Test | 98.25% |

**Gaps:** Train-Val: 0.18%, Train-Test: 0.28%

**Diagnosis:** No overfitting, no underfitting — good generalization. Model complexity (Logistic Regression) is appropriate for this dataset size.

### Learning Curves Interpretation
- Training accuracy stayed consistently high (~99%) across all training sizes
- Validation accuracy converged to ~96-97% as training set grew
- Curves trending toward convergence → adding more data could still help slightly, but diminishing returns

### Error Analysis Applied
- Total test errors: 2 out of 114 (1.75% error rate)
- Error 1: Benign misclassified as Malignant (False Positive)
- Error 2: Malignant misclassified as Benign (False Negative — clinically more serious)
- **No systematic error pattern detected** — errors appear to be isolated borderline cases

### Model Selection Process
1. Trained 3 models on training set
2. Evaluated all 3 on validation set → Logistic Regression best (98.35% val accuracy)
3. Reported final performance on test set (one time only) → 98.25% test accuracy

### Statistical Testing Beyond Standard Evaluation
McNemar's test compared Logistic Regression vs. SVM:
- Both models agreed on 111/114 samples
- p-value = 1.0 → difference NOT statistically significant

**Lesson:** Numerical difference in accuracy (98.25% vs 97.37%) does not mean one model is meaningfully better than the other at this sample size.

---

## 11. Key Insights & Open Questions

### Key Insights
- Three-set split (train/val/test) is not optional — it is the correct methodology
- Bias/variance diagnosis must come before choosing a fix — wrong diagnosis = wasted effort
- Error analysis by manual inspection gives direction that aggregate metrics cannot
- More data helps high-variance models; it does not help high-bias models
- Statistical significance testing (McNemar's) revealed that numerical accuracy differences can be misleading

### Lessons Learned from the Project
- Stratified splitting was essential — without it, random chance could over/under-represent minority class
- Feature scaling (StandardScaler) had a measurable positive effect on Logistic Regression and SVM convergence
- Default hyperparameters were already strong; GridSearch confirmed LR baseline was near-optimal
- Small test set (114 samples) limits statistical power — even 1-2 predictions changing accuracy by ~1%

### Open Questions for Further Study
- How does the choice of K in cross-validation affect model selection stability?
- Can we always trust learning curves with such small datasets (high variance in the curves themselves)?
- When is it appropriate to combine train+val sets and retrain on all data before final test?
- What is the theoretical relationship between training set size and generalization gap?

---

## References & Resources
- Andrew Ng's ML Course: Advanced Learning Algorithms — Evaluation & Model Development
- Scikit-learn Documentation: Model Evaluation, Cross-Validation
- Hands-On Machine Learning (Aurélien Géron): Chapter 2 (End-to-End ML Project)
- Pattern Recognition and Machine Learning (Bishop): Chapter 3 (Linear Models)
