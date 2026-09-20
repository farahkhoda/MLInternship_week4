# Supervised Machine Learning: Regression and Classification
## Study Notes — Week 4

---

## 1. Linear Regression

### Model Representation
**Hypothesis function:** h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

In vectorized form: h(x) = θᵀX

Each parameter θ represents the weight/slope for each feature.

### Cost Function (Mean Squared Error)
J(θ) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

Where m = number of training samples, (x⁽ⁱ⁾, y⁽ⁱ⁾) = i-th training example

Goal: Find parameters θ that minimize J(θ)

### Gradient Descent
Update rule (simultaneously for all j):

θⱼ := θⱼ - α × ∂J(θ)/∂θⱼ

Where α = learning rate, ∂J(θ)/∂θⱼ = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾

Repeat until convergence (cost stops decreasing significantly).

### Feature Scaling
**Why:** Features with very different scales (e.g., size 0-1000, price 0-100) can slow gradient descent.

**Method - Standardization (Z-score normalization):**
x_scaled = (x - mean) / std_dev

**Method - Min-Max Normalization:**
x_scaled = (x - min) / (max - min)

**Benefit:** Brings all features to roughly [-1, 1] range; gradient descent converges faster.

### Learning Rate Selection
- **Too small α:** Converges very slowly
- **Too large α:** May overshoot the minimum; cost might not decrease
- **Good α:** Cost decreases consistently; convergence in reasonable iterations
- **Typical values:** Start with 0.01, 0.1, 1, 10, etc.; try different values and plot cost vs. iterations

---

## 2. Multiple Linear Regression

### Vectorization
Instead of loops, use matrix operations:

```python
# Non-vectorized (slow)
prediction = 0
for j in range(n):
    prediction += theta[j] * x[j]

# Vectorized (fast)
prediction = np.dot(theta, x)  # or theta.T @ x
```

Benefits: Faster computation, cleaner code, leverages optimized libraries (NumPy, etc.)

### Feature Engineering
**Creating new features from existing ones:**
- If data shows non-linear pattern, create polynomial features: x₂ = x₁², x₃ = x₁³
- Combine features: x_area = length × width
- Use domain knowledge to craft meaningful features

**Example:** In medical diagnosis (Breast Cancer dataset), "worst texture" is already engineered from raw pixel data

### Polynomial Regression
Transform linear regression into polynomial form:

For degree-2 polynomial: h(x) = θ₀ + θ₁x + θ₂x²

Treat x² as a new feature and apply linear regression.

**Trade-off:** Higher degree captures complexity but risks overfitting.

---

## 3. Classification with Logistic Regression

### Why Not Linear Regression for Classification
- Linear regression predicts continuous values; can output any value
- For binary classification, we need output in [0, 1] (probability)
- Linear model can produce predictions > 1 or < 0, which don't make sense as probabilities
- **Solution:** Use logistic regression with sigmoid function to squash output to [0, 1]

### Sigmoid Function
σ(z) = 1 / (1 + e^(-z))

**Properties:**
- Output always between 0 and 1
- σ(0) = 0.5
- σ(large positive z) → 1
- σ(large negative z) → 0

In logistic regression: z = θᵀX, so h(x) = σ(θᵀX)

### Decision Boundary
**Prediction rule:**
- If h(x) ≥ 0.5 → predict class 1
- If h(x) < 0.5 → predict class 0

**Decision boundary** = set of points where h(x) = 0.5

For linear model (without polynomial features), decision boundary is a line/plane.

### Logistic Loss / Cost Function
**Binary cross-entropy loss:**

J(θ) = -(1/m) × Σ[y⁽ⁱ⁾ × log(h(x⁽ⁱ⁾)) + (1 - y⁽ⁱ⁾) × log(1 - h(x⁽ⁱ⁾))]

**Why this?**
- If y = 1 and h(x) = 1: loss = 0 (correct prediction)
- If y = 1 and h(x) → 0: loss → ∞ (heavily penalizes wrong prediction)
- Equivalent interpretation: Maximum likelihood estimation for Bernoulli distribution

### Gradient Descent for Logistic Regression
Update rule (same form as linear regression!):

θⱼ := θⱼ - α × ∂J(θ)/∂θⱼ

Where ∂J(θ)/∂θⱼ = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾

(Coincidentally, the derivative has the same form as linear regression, but h(x) is now sigmoid, not linear.)

---

## 4. Overfitting & Regularization

### Underfitting vs. Overfitting

**Underfitting (High Bias):**
- Model too simple to capture underlying pattern
- High training error, high test error
- Poor fit to both training and test data
- Example: Trying to fit a curved dataset with a straight line

**Overfitting (High Variance):**
- Model memorizes training data, including noise
- Low training error, high test error
- Perfect fit to training data but poor generalization
- Example: Decision tree with no depth limit on small dataset

### Addressing Overfitting

**More data:**
- Collect larger dataset; more diverse data helps model generalize better
- Dilutes effect of noise in training set

**Feature selection:**
- Remove irrelevant or redundant features
- Reduces model complexity
- Simpler model is less likely to overfit

**Regularization:**
- Penalize large parameter values during training
- Discourages model from relying too heavily on any single feature
- Smooth decision boundary; less prone to fitting noise

### Regularized Linear Regression (Ridge / L2)
**Cost function:**

J(θ) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ/2m) × Σθⱼ²

**Effect of λ:**
- λ = 0: No regularization (original cost)
- λ very large: Encourages all θⱼ → 0; model approaches mean prediction (high bias)
- λ moderate: Sweet spot balancing fit and complexity

**Note:** Usually don't regularize θ₀ (bias term)

### Regularized Logistic Regression
**Cost function:**

J(θ) = -(1/m) × Σ[y⁽ⁱ⁾ × log(h(x⁽ⁱ⁾)) + (1 - y⁽ⁱ⁾) × log(1 - h(x⁽ⁱ⁾))] + (λ/2m) × Σθⱼ²

Same principle: Regularization term penalizes large coefficients.

---

## 5. Evaluation Metrics (Classification)

### Accuracy, Precision, Recall, F1

**Accuracy** = (TP + TN) / Total

Fraction of correct predictions. Good when classes are balanced.

**Precision** = TP / (TP + FP)

"Of all predicted positive, how many were actually positive?" Important when false positives are costly.

**Recall (Sensitivity)** = TP / (TP + FN)

"Of all actual positives, how many did we catch?" Important when false negatives are costly (e.g., disease detection).

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

Harmonic mean of precision and recall; single metric balancing both.

### Confusion Matrix
```
                Predicted
                Pos  Neg
Actual Pos      TP   FN
       Neg      FP   TN
```

- **TP (True Positive):** Predicted positive, actually positive — correct
- **TN (True Negative):** Predicted negative, actually negative — correct
- **FP (False Positive):** Predicted positive, actually negative — wrong (Type I error)
- **FN (False Negative):** Predicted negative, actually positive — wrong (Type II error)

### ROC & AUC
**ROC Curve:** Plots True Positive Rate (TPR = Recall) vs. False Positive Rate (FPR = FP / (FP + TN))

Computed by varying the classification threshold from 0 to 1.

**AUC (Area Under Curve):** Single number summarizing ROC curve (0 to 1)
- 0.5 = random classifier
- 1.0 = perfect classifier
- Higher AUC = better model

**Use:** Especially useful for imbalanced datasets.

---

## 6. Feature Engineering & Selection

### Polynomial Features
Transform x → [x, x², x³, ...] to capture non-linear relationships.

**Drawback:** Increasing polynomial degree exponentially increases feature space; risk of overfitting.

### Feature Normalization
Covered in Section 1 (Feature Scaling). Essential before training most algorithms.

### Feature Selection Methods
**Manual:** Domain knowledge; remove features known to be irrelevant.

**Univariate:** For each feature, measure correlation with target; keep top-k features.

**Recursive Feature Elimination (RFE):** Train model, remove least important feature, retrain; repeat.

**Lasso Regression:** Regularization naturally drives less important features' coefficients to zero.

---

## 7. Connection to This Week's Side Quest: Breast Cancer Classifier

### How Logistic Regression Was Applied

In the Breast Cancer dataset project:

1. **Feature Scaling:** 30 features had very different scales (e.g., radius 6-28, smoothness 0.05-0.16). Applied `StandardScaler` to normalize all features to mean ≈ 0, std ≈ 1. This enabled fair learning across all features.

2. **Logistic Regression Model:** Built a binary classifier using logistic regression:
   - Input: 30 normalized features (measurements of cell nuclei)
   - Output: Probability of malignancy (0 = benign, 1 = malignant)
   - Used sigmoid function to map predictions to [0, 1]
   - Trained with gradient descent on binary cross-entropy loss

3. **Decision Boundary:** The trained model learned a (30-dimensional) hyperplane separating benign from malignant tumors in feature space.

4. **Evaluation:** Accuracy 98.25%, Precision 98.61%, Recall 98.61%, F1 0.9861 on test set — showing the model learned a good decision boundary.

### Overfitting Analysis

**Bias/Variance check in the project:**
- Training accuracy: 98.53%
- Validation accuracy: 98.35%
- Test accuracy: 98.25%

**Conclusion:** Gap of 0.18-0.28% is extremely small; **no overfitting detected**. The model generalizes well.

**Why no overfitting despite 30 features?**
- Dataset is balanced (62.7% benign, 37.3% malignant)
- 569 samples is sufficient relative to feature count
- Logistic regression is inherently regularized by its sigmoid formulation (outputs probability, not raw unbounded values)
- 3-way split (train/val/test) ensured held-out test data was never seen during training

### Feature Importance Discovered

Logistic regression coefficients revealed which features matter most:
1. worst texture (coefficient ≈ 1.06)
2. radius error (coefficient ≈ 0.98)
3. worst radius (coefficient ≈ 0.96)

These "worst" measurements (extreme values) are stronger indicators of malignancy than mean measurements — aligns with medical intuition.

---

## 8. Key Insights & Open Questions

### What Worked Well
- Logistic regression's simplicity and interpretability made it ideal for this binary classification task
- Feature normalization was crucial; raw features had vastly different scales
- Train/val/test split with stratification preserved class distribution, preventing class imbalance artifacts

### Lessons Learned
- Accuracy alone can be misleading; looked at precision, recall, F1, confusion matrix, and ROC/AUC for complete picture
- Statistical significance testing (McNemar's test) showed that LR's numerical advantage over SVM wasn't statistically significant — number of samples matters
- Cross-validation (5-fold) provided robust estimate of generalization; individual test runs can have variance

### Open Questions for Future Work
- Could ensemble methods (Random Forest, Gradient Boosting) improve beyond 98.25%?
- Would feature selection reduce dimensionality without sacrificing accuracy?
- How would the model perform on an external dataset from a different hospital?
- What if we applied polynomial features (x²) to capture non-linear relationships?
- Could we tune the decision threshold to optimize for recall (catch more cancers, accept more false positives)?

---

## References & Resources
- Andrew Ng's ML Course: Supervised Learning, Logistic Regression
- Scikit-learn Documentation: Linear Models
- Hands-On Machine Learning (Aurélien Géron): Chapters 4-5
- UCI Breast Cancer Wisconsin Dataset Documentation
