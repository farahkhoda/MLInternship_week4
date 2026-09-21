MLInternship_week4

Evaluate the Classical Classifier — Week 4 Internship Project at Darnica

⸻

Project Overview

This repository contains my Week 4 ML internship work at Darnica.

The project follows two parallel tracks:

Main Quest

Supervised Machine Learning (Regression & Classification) + Advanced Learning Algorithms

The main focus of this week was learning and applying rigorous model evaluation and development practices, including train/validation/test methodology, bias-variance analysis, error analysis, and generalization.

Main Practical Project — YCB-Video

Evaluate a Classical Classifier on YCB-Video

The primary practical project for Week 4 applies these concepts to the YCB-Video dataset.

A classical image classifier was developed using handcrafted RGB and geometric features extracted from object masks. The main emphasis was on evaluating the classifier under a realistic unseen-scene generalization setting, rather than relying only on random sample-level splits.

Extra Practice Project — Breast Cancer Wisconsin

Binary Classification on Medical Data

As an additional exercise beyond the main YCB-Video assignment, I independently applied the same evaluation methodology to the Breast Cancer Wisconsin dataset.

This project was used as a practice environment to strengthen my understanding of classification, model comparison, evaluation metrics, error analysis, and bias-variance diagnosis.

⸻

1. Main Quest — Supervised Learning & Advanced Learning Algorithms

The theoretical component of Week 4 focused on supervised learning and advanced model-development strategies.

Supervised Learning: Regression & Classification

Topics studied:

* Linear Regression: Cost function (MSE), gradient descent, feature scaling, and learning rate selection
* Logistic Regression: Sigmoid function, decision boundaries, binary cross-entropy loss, and why logistic regression is appropriate for classification
* Regularization: L1 (Lasso) and L2 (Ridge) techniques for reducing overfitting and controlling model complexity
* Feature Engineering: Polynomial features, normalization, and feature selection
* Evaluation Metrics: Accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and AUC-ROC

Key Insight

Different models and evaluation metrics are appropriate for different problems. Reliable model development requires understanding both the learning algorithm and the evaluation methodology.

⸻

2. Advanced Learning Algorithms — Evaluation & Model Development

The Week 4 material also focused heavily on evaluating and developing machine learning models correctly.

Topics studied:

* Train/Validation/Test Split: Separating model development from final evaluation
* Data Leakage: Understanding how information from validation or test data can incorrectly influence training
* Feature Scaling: Fitting preprocessing steps only on training data
* Bias/Variance Tradeoff: Understanding underfitting and overfitting
* Generalization: Evaluating whether a model performs well on previously unseen data
* Learning Curves: Comparing training and validation performance
* Error Analysis: Investigating systematic patterns in model failures
* Model Selection: Using validation data for development and reserving the test set for final evaluation
* Classical Algorithms: Logistic Regression, Decision Trees, SVM, KNN, and ensemble methods

Key Insight

A high training or validation score does not automatically mean that a model generalizes well.

A rigorous evaluation pipeline is necessary to distinguish between fitting the available data and learning patterns that remain useful on unseen data.

⸻

3. Main Practical Project — YCB-Video

Project: Evaluate a Classical Classifier

The primary practical project for Week 4 was implemented using the YCB-Video dataset.

The goal was to build a classical object classifier and evaluate not only its in-sample performance, but also its ability to generalize to a completely unseen scene.

⸻

Dataset

The experiment used 12 YCB-Video test scenes:

000048
000049
000050
000051
000052
000053
000054
000055
000056
000057
000058
000059

The selected subset contained:

* 4,125 object instances
* 12 scenes
* 21 object classes

The target variable was:

obj_id

representing the YCB object class.

⸻

4. Feature Engineering

Instead of using raw images directly, handcrafted features were extracted from each object’s RGB image and visible segmentation mask.

RGB Features

The following color statistics were calculated:

* mean_r
* mean_g
* mean_b
* std_r
* std_g
* std_b

These describe the average color and color variation within the object region.

Geometric Features

The following shape and size features were calculated:

* area_ratio
* bbox_width
* bbox_height
* aspect_ratio

These describe the approximate size and geometry of the visible object.

The final feature vector contained 10 handcrafted features.

⸻

5. Initial Sample-Level Experiment

An initial experiment was performed on Scene 000048.

The scene contained five object classes:

1, 6, 14, 19, 20

with 375 samples.

The data was split using a stratified train/validation/test split:

Train       → 225
Validation  → 75
Test        → 75

A StandardScaler was fitted only on the training set and then used to transform the validation and test sets.

A Logistic Regression classifier was trained using:

LogisticRegression(
    max_iter=1000,
    random_state=42
)

The model achieved 100% accuracy on the train, validation, and test subsets.

However, this result had an important limitation:

All three subsets came from the same scene.

Therefore, the result did not provide strong evidence that the classifier could generalize to a completely unseen YCB-Video scene.

This observation motivated a more rigorous unseen-scene evaluation.

⸻

6. Unseen-Scene Evaluation

For the main evaluation, Scene 000048 was completely held out as the final unseen test scene.

The model was trained using other scenes containing the same five target classes:

1, 6, 14, 19, 20

The resulting training pool contained:

675 samples

Class distribution:

Object 1   → 225
Object 6   → 225
Object 14  → 75
Object 19  → 75
Object 20  → 75

The training pool was then divided into:

Training      → 540 samples
Validation    → 135 samples
Unseen Test   → 375 samples

The unseen test scene contained 75 samples from each target class.

⸻

7. Data Leakage Prevention

The scaler was fitted exclusively on the training data:

scaler.fit(X_train)

and subsequently applied to validation and test data:

scaler.transform(X_val)
scaler.transform(X_test)

This prevents information from the validation and unseen test sets from influencing the preprocessing stage.

The scaled training features had approximately zero mean and unit variance.

⸻

8. Model

The main classifier used in this experiment was:

Logistic Regression

Configuration:

LogisticRegression(
    max_iter=1000,
    random_state=42
)

The model was trained on the 10 standardized handcrafted features.

⸻

9. Results

Train and Validation

The model achieved:

Dataset	Accuracy
Train	100%
Validation	100%

The train-validation gap was:

0 percentage points

This indicates that the model was able to fit the training data and achieved equally strong performance on the in-pool validation subset.

⸻

Unseen Scene

When evaluated on the completely unseen Scene 000048:

Correct predictions: 215 / 375
Incorrect predictions: 160 / 375
Accuracy: 57.3%
Error rate: 42.7%

The resulting performance gap was:

Train       → 100%
Validation  → 100%
Unseen Test → 57.3%

This demonstrates that strong performance on samples drawn from the training pool did not translate into equally strong scene-level generalization.

⸻

10. Classification Report

The unseen-scene classification results were:

Class	Precision	Recall	F1-score	Support
Object 1	0.51	0.41	0.46	75
Object 6	0.80	1.00	0.89	75
Object 14	0.43	0.75	0.54	75
Object 19	0.00	0.00	0.00	75
Object 20	0.79	0.71	0.75	75
Accuracy			0.57	375
Macro Avg	0.50	0.57	0.53	375
Weighted Avg	0.50	0.57	0.53	375

Important observations

Object 6 achieved the strongest class-level result, with:

Recall = 1.00
F1 = 0.89

All 75 Object 6 samples were correctly classified.

Object 19 was the most difficult class:

Precision = 0.00
Recall = 0.00
F1 = 0.00

None of the 75 Object 19 samples were correctly classified in the unseen scene.

⸻

11. Confusion Matrix

The unseen-scene confusion matrix was:

                 Predicted
              1   6   14  19  20
Actual  1    31   0   44   0   0
        6     0  75    0   0   0
       14     0  19   56   0   0
       19    30   0   31   0  14
       20     0   0    0  22  53

The most frequent confusion patterns were:

Object 1  → Object 14 : 44
Object 19 → Object 14 : 31
Object 19 → Object 1  : 30
Object 20 → Object 19 : 22
Object 14 → Object 6  : 19
Object 19 → Object 20 : 14

These errors were concentrated among specific object classes rather than being uniformly distributed.

⸻

12. Error Analysis

The model made:

160 errors out of 375 unseen-scene samples.

The largest error pattern was:

Object 1 → Object 14
44 samples

Object 19 showed the most severe failure pattern:

Object 19 → Object 14 : 31
Object 19 → Object 1  : 30
Object 19 → Object 20 : 14
Object 19 → Object 19 : 0

This indicates that the current handcrafted feature representation does not provide sufficient separation between Object 19 and several other classes under the visual conditions of Scene 000048.

The strong confusion between Objects 1 and 14 also suggests substantial overlap between their representations in the current feature space.

⸻

13. Bias, Variance & Generalization

Underfitting

There is no strong evidence of underfitting.

The classifier achieved 100% training accuracy, demonstrating that it was capable of fitting the training data.

Overfitting

The results are consistent with sensitivity to scene-specific feature distributions and limited scene-level generalization.

The most important evidence is the large performance difference:

Train       → 100%
Validation  → 100%
Unseen Test → 57.3%

However, this should not be described as definitive proof of overfitting because the validation split was sample-level rather than fully scene-independent.

Bias

The high training accuracy provides little evidence of high bias in this experiment.

Variance

The substantial performance decrease on the unseen scene is consistent with increased sensitivity to distribution changes and suggests limited robustness.

Generalization

The classifier showed limited scene-level generalization.

This was the most important finding of the practical experiment.

⸻

14. Methodological Limitation

A fully scene-independent train/validation split containing all five target classes was not possible within the selected scenes because some target classes appeared in only one training scene.

Therefore, the evaluation should be described precisely as:

Sample-level train/validation evaluation with a completely held-out unseen-scene test.

The unseen-scene test provides the strongest evidence about scene-level generalization in this experiment, while the validation score should not be interpreted as an independent scene-level generalization estimate.

⸻

15. Main Findings

The YCB-Video experiment demonstrated several important lessons:

1. Handcrafted RGB and geometric features can provide strong classification performance on data drawn from familiar scenes.
2. A perfect train/validation score does not necessarily imply strong generalization.
3. Holding out an entire scene revealed a substantial generalization gap.
4. The unseen-scene accuracy was 57.3%.
5. Error analysis showed that failures were concentrated among particular object classes.
6. Object 19 was particularly difficult for the current feature representation.
7. Object 6 was classified correctly in all 75 unseen-scene samples.
8. The results suggest that the current handcrafted features are sensitive to scene-specific visual distributions.
9. More robust models and richer feature representations should be investigated in future experiments.

⸻

16. Extra Practice Project — Breast Cancer Wisconsin

In addition to the required YCB-Video practical project, I completed a separate classification experiment using the Breast Cancer Wisconsin dataset.

This was an independent practice project designed to reinforce the evaluation concepts from the Week 4 Main Quest before applying them to YCB-Video.

Task

Classify breast tumors as:

* Malignant
* Benign

using 30 numerical diagnostic features.

Dataset

The dataset contained:

* 569 samples
* 30 numerical features
* 2 classes
* Approximately 62.7% benign
* Approximately 37.3% malignant

⸻

17. Breast Cancer Project Approach

The practice project included:

1. Exploratory Data Analysis
2. Data quality checks
3. Feature scaling
4. Stratified train/validation/test splitting
5. Logistic Regression
6. Decision Tree
7. Support Vector Machine
8. Accuracy, precision, recall, and F1 evaluation
9. Confusion matrices
10. ROC/AUC analysis
11. Cross-validation
12. Learning curves
13. Error analysis
14. Bias/variance analysis

⸻

18. Breast Cancer Results

The Logistic Regression model achieved:

Test Accuracy: 98.25%
Precision:      98.61%
Recall:         98.61%
F1-score:       98.61%
AUC-ROC:        0.9957

There were only:

2 misclassifications
out of 114 test samples

The training-test performance gap was approximately:

0.29 percentage points

The results showed strong generalization within this experimental setup.

⸻

19. Additional Breast Cancer Exploration

The practice project also included several experiments beyond the basic Week 4 requirements.

Hyperparameter Tuning

GridSearchCV was used to explore Decision Tree and SVM hyperparameters.

Decision Tree F1-score improved from:

0.9296 → 0.9362

The SVM experiments also demonstrated that the model with the best cross-validation result does not necessarily have the best final test performance.

Lesson

Cross-validation helps with model selection, but the test set should remain untouched until final evaluation.

⸻

Precision-Recall Curves

Precision-Recall curves were calculated alongside ROC curves.

For the medical classification task, Precision-Recall analysis provided an additional perspective because false-negative errors are particularly important in diagnostic classification.

Logistic Regression achieved:

Average Precision = 0.9973

⸻

McNemar’s Test

McNemar’s statistical test was used to compare Logistic Regression and SVM predictions.

The result was:

p-value = 1.0

This did not provide evidence of a statistically significant difference between the two models under this test.

Lesson

A small numerical difference in test accuracy does not automatically imply a meaningful difference in model performance.

⸻

20. Key Lessons from Week 4

1. Evaluation methodology is critical

A model can appear perfect under an easy split while performing substantially worse on truly unseen data.

2. Test data must remain independent

The final test set should not influence model training, preprocessing, or model selection.

3. Scaling can introduce leakage

Preprocessing transformations such as StandardScaler should be fitted only on the training data.

4. Accuracy is not enough

Accuracy should be complemented with:

* Precision
* Recall
* F1-score
* Confusion matrix
* Error analysis

5. Error analysis reveals model behavior

Looking at which classes are confused provides more information than a single accuracy value.

6. Generalization matters

The YCB-Video experiment demonstrated that strong in-pool performance does not necessarily translate to strong performance on a new scene.

7. More complex models are not automatically better

Model selection should be based on appropriate validation methodology and generalization performance rather than complexity alone.

8. Different datasets can expose different problems

The Breast Cancer practice project showed strong generalization under its evaluation setup, while the YCB-Video experiment exposed a significant scene-level generalization challenge.

⸻

21. Deliverables

Main YCB-Video Project

* classifier_evaluation.ipynb — complete YCB-Video classification and evaluation pipeline
* weekly_report_week4.md — detailed Week 4 methodology, results, and analysis
* report.md — project-level summary
* YCB-Video scene data used for the experiment
* Feature extraction pipeline
* Classification metrics
* Confusion matrix
* Error analysis
* Generalization analysis

Main Quest Study Materials

* regression_classification_notes.md
* advanced_learning_algorithms_notes.md

These contain notes covering supervised learning, evaluation methodology, bias/variance, and classical machine learning algorithms.

Extra Practice Project

* Breast Cancer Wisconsin classification notebook
* EDA and preprocessing analysis
* Model comparison
* ROC/AUC and Precision-Recall analysis
* Learning curves
* Hyperparameter tuning
* McNemar’s statistical test

⸻

22. Skills Demonstrated

Machine Learning

✅ Supervised learning
✅ Logistic Regression
✅ Classification
✅ Feature engineering
✅ Feature scaling
✅ Train/validation/test methodology
✅ Stratified splitting
✅ Model evaluation

Evaluation & Generalization

✅ Accuracy, precision, recall, and F1-score
✅ Confusion matrix analysis
✅ Error analysis
✅ Bias/variance diagnosis
✅ Overfitting and underfitting analysis
✅ Scene-level generalization testing
✅ Data leakage prevention

Data & Computer Vision

✅ YCB-Video dataset handling
✅ RGB image processing
✅ Object segmentation masks
✅ Image-derived feature extraction
✅ Geometric feature engineering

Statistical & Experimental Practice

✅ Cross-validation
✅ Hyperparameter tuning
✅ ROC/AUC analysis
✅ Precision-Recall analysis
✅ Statistical significance testing

Engineering & Documentation

✅ Reproducible ML workflow
✅ Jupyter-based experimentation
✅ Structured project organization
✅ Technical reporting
✅ Documentation of limitations
✅ Evidence-based model analysis

⸻

23. Conclusion

Week 4 focused on an important transition from simply training machine learning models to evaluating whether their results can actually be trusted.

The main practical project used YCB-Video to evaluate a classical Logistic Regression classifier under an unseen-scene setting.

Although the model achieved 100% accuracy on the training and in-pool validation data, its accuracy decreased to 57.3% on the completely unseen Scene 000048.

Rather than treating this result as simply a low score, the experiment used it to investigate the model’s behavior through classification metrics, confusion matrices, error analysis, and bias-variance/generalization analysis.

The experiment demonstrated that:

High performance on familiar data does not necessarily imply strong generalization to unseen data.

The Breast Cancer Wisconsin project was completed separately as an additional practice exercise and helped reinforce the evaluation concepts before and alongside their application to the primary YCB-Video project.

The YCB-Video results provide a baseline for the next stage of the internship, where different classical algorithms can be compared using the same feature representation.

⸻

Next Step

The next stage will investigate additional classical models, including:

Logistic Regression
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Boosted Trees

The goal will be to determine whether different model families can improve classification performance and generalization on the same YCB-Video feature set.