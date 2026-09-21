Week 4 — YCB-Video Classical Classification

Model Evaluation, Error Analysis, and Scene-Level Generalization

⸻

1. Project Overview

This week, the classical image classification experiment was extended from a single-scene experiment to a larger subset of the YCB-Video dataset.

The goal was not only to train a classical classifier, but also to evaluate its performance carefully using separate training, validation, and test data, analyze classification errors, and investigate whether the model generalizes to a completely unseen scene.

A Logistic Regression classifier was trained using handcrafted RGB and geometric image features extracted from object masks.

The main focus of this experiment was:

* Train/Validation/Test splitting
* Preventing data leakage during scaling
* Classical model training
* Evaluation using multiple classification metrics
* Confusion matrix analysis
* Error analysis
* Bias/variance observations
* Unseen-scene generalization

⸻

2. Dataset

The experiment used 12 YCB-Video test scenes:

* 000048
* 000049
* 000050
* 000051
* 000052
* 000053
* 000054
* 000055
* 000056
* 000057
* 000058
* 000059

The combined dataset contained:

* 4,125 object instances
* 12 scenes
* 21 object classes

The number of samples per scene varied between 225 and 450.

The target variable was:

obj_id

which represents the YCB object class.

⸻

3. Feature Engineering

Instead of using raw images directly, handcrafted features were extracted from each object’s RGB image and visible segmentation mask.

The following 10 features were used:

RGB features

* mean_r
* mean_g
* mean_b
* std_r
* std_g
* std_b

These describe the average color and color variation inside the object region.

Geometric features

* area_ratio
* bbox_width
* bbox_height
* aspect_ratio

These describe the approximate size and shape of the detected object.

The feature vector for each sample was therefore:

[mean_r, mean_g, mean_b,
 std_r, std_g, std_b,
 area_ratio,
 bbox_width, bbox_height,
 aspect_ratio]

⸻

4. Initial Single-Scene Experiment

An initial experiment was performed using Scene 000048.

This scene contained five object classes:

1, 6, 14, 19, 20

There were 375 samples in total.

The data was split into:

* 225 training samples
* 75 validation samples
* 75 test samples

The split was stratified by object class.

A StandardScaler was fitted only on the training data and then used to transform the validation and test sets.

A Logistic Regression classifier was trained using:

LogisticRegression(
    max_iter=1000,
    random_state=42
)

The model achieved 100% accuracy on the train, validation, and test subsets.

However, this result had an important limitation: all three subsets came from the same scene.

Therefore, this experiment did not provide strong evidence of generalization to unseen scenes.

⸻

5. Unseen-Scene Evaluation

To obtain a more meaningful estimate of generalization, Scene 000048 was completely held out as an unseen test scene.

The model was trained using samples from other scenes containing the same five target classes:

1, 6, 14, 19, 20

The training pool contained 675 samples.

Its class distribution was:

Object 1   → 225
Object 6   → 225
Object 14  → 75
Object 19  → 75
Object 20  → 75

Because some classes appeared in only one training scene, a fully scene-independent train/validation split containing every class was not possible.

Therefore:

* Scene 000048 was kept completely unseen as the final test set.
* The remaining training pool was split into train and validation samples using stratification.

The resulting split was:

Training      → 540 samples
Validation    → 135 samples
Unseen Test   → 375 samples

The unseen test set contained 75 samples from each of the five classes.

⸻

6. Data Scaling

Standardization was performed using StandardScaler.

Importantly, the scaler was fitted only on the training data:

scaler.fit(X_train)

and then applied to validation and test data:

scaler.transform(X_val)
scaler.transform(X_test)

This prevents information from the validation or unseen test set from influencing the preprocessing stage.

The scaled training features had approximately:

Mean ≈ 0
Standard deviation ≈ 1

This confirmed that the scaler was correctly fitted to the training set.

⸻

7. Model Training

A Logistic Regression classifier was trained on the scaled training features.

The model successfully trained on:

540 training samples
10 input features
5 object classes

Predictions were generated separately for:

* Training set
* Validation set
* Unseen Scene 000048

⸻

8. Train and Validation Performance

The model achieved:

Dataset	Accuracy
Train	100%
Validation	100%

The train-validation accuracy gap was:

0 percentage points

This means that the model was able to fit the training data extremely well and also performed perfectly on the in-pool validation subset.

There was no evidence of classical underfitting in these results.

⸻

9. Unseen-Scene Test Performance

When evaluated on the completely unseen Scene 000048, performance dropped substantially.

Results:

Correct predictions: 215 / 375
Incorrect predictions: 160 / 375
Accuracy: 57.3%
Error rate: 42.7%

Therefore:

Train Accuracy       = 100%
Validation Accuracy  = 100%
Unseen Test Accuracy = 57.3%

This large performance gap demonstrates that performance on samples from the training pool does not necessarily translate to strong performance on a new scene.

The result indicates limited scene-level generalization and is consistent with sensitivity to scene-specific feature distributions.

⸻

10. Classification Report

The detailed classification results on the unseen scene were:

Class	Precision	Recall	F1-score	Support
Object 1	0.51	0.41	0.46	75
Object 6	0.80	1.00	0.89	75
Object 14	0.43	0.75	0.54	75
Object 19	0.00	0.00	0.00	75
Object 20	0.79	0.71	0.75	75
Accuracy			0.57	375
Macro Avg	0.50	0.57	0.53	375
Weighted Avg	0.50	0.57	0.53	375

Per-class observations

Object 6

Object 6 had the strongest recall:

Recall = 1.00

All 75 Object 6 samples were correctly identified.

Object 19

Object 19 had the weakest performance:

Precision = 0.00
Recall = 0.00
F1 = 0.00

None of the 75 Object 19 samples were classified correctly.

Object 1

Only 31 of 75 Object 1 samples were correctly classified:

Recall ≈ 41%

A large number were classified as Object 14.

Object 14

Object 14 achieved relatively high recall but lower precision, indicating that the model identified many real Object 14 samples but also incorrectly predicted Object 14 for other classes.

Object 20

Object 20 achieved:

Precision = 0.79
Recall = 0.71
F1 = 0.75

indicating moderate performance with some confusion with Object 19.

⸻

11. Confusion Matrix Analysis

The confusion matrix for the unseen scene was:

                 Predicted
              1   6   14  19  20
Actual  1    31   0   44   0   0
        6     0  75    0   0   0
       14     0  19   56   0   0
       19    30   0   31   0  14
       20     0   0    0  22  53

The major confusion patterns were:

Object 1  → Object 14 : 44
Object 19 → Object 14 : 31
Object 19 → Object 1  : 30
Object 20 → Object 19 : 22
Object 14 → Object 6  : 19
Object 19 → Object 20 : 14

These errors were concentrated around a small number of class pairs rather than being uniformly distributed across all classes.

⸻

12. Error Analysis

A total of 160 errors occurred on the unseen scene.

The most frequent error was:

Object 1 → Object 14
44 samples

Object 19 was the most problematic class.

Its 75 samples were distributed as:

Object 19 → Object 14 : 31
Object 19 → Object 1  : 30
Object 19 → Object 20 : 14
Object 19 → Object 19 : 0

Therefore, the model did not correctly identify any Object 19 samples in the unseen scene.

This suggests that the current handcrafted feature representation does not provide sufficient separation between Object 19 and several other classes under the conditions present in Scene 000048.

Similarly, the strong confusion between Objects 1 and 14 indicates overlap in the feature space used by the classifier.

⸻

13. Bias-Variance and Generalization Analysis

Underfitting

There is no strong evidence of underfitting.

The model achieved 100% training accuracy, meaning it was capable of fitting the training data.

Overfitting

The results are consistent with overfitting or sensitivity to scene-specific feature distributions.

The main evidence is the large gap:

Train       → 100%
Validation  → 100%
Unseen Test → 57.3%

However, the results should not be interpreted as definitive proof of overfitting because the validation split was sample-level rather than fully scene-independent.

Bias

The high training performance provides little evidence of high bias in this experiment.

Variance

The large decrease in performance on an unseen scene is consistent with high sensitivity to changes in the data distribution and therefore suggests limited robustness.

Generalization

The model demonstrated weak scene-level generalization.

This is particularly important because YCB-Video contains multiple scenes with different visual conditions, object arrangements, and image distributions.

⸻

14. Important Methodological Limitation

The training pool did not contain every target class across multiple independent scenes.

In particular, Objects 14, 19, and 20 each appeared in only one training scene within the selected pool.

Therefore, it was not possible to construct a completely scene-independent train/validation split while keeping all five target classes represented in both sets.

For this reason, the experiment should be described as:

Sample-level train/validation evaluation with a completely held-out unseen-scene test.

The unseen-scene test is therefore the strongest part of the generalization evaluation, while the validation score should not be interpreted as an independent scene-level generalization estimate.

⸻

15. Main Findings

The main findings of this experiment were:

1. Handcrafted RGB and geometric features were sufficient for very strong performance on the in-pool train and validation data.
2. The same model achieved only 57.3% accuracy on a completely unseen scene.
3. Therefore, high in-pool accuracy did not translate into strong scene-level generalization.
4. Error analysis showed that the mistakes were concentrated among specific object classes.
5. Object 19 was particularly difficult for the current feature representation.
6. Object 6 was consistently recognized in the unseen scene.
7. The results suggest that the current feature representation is sensitive to scene-specific visual distributions.
8. More robust models and/or richer feature representations should be investigated.

⸻

16. Conclusion

This experiment demonstrated the importance of evaluating machine learning models beyond a single train/test split.

The initial 100% accuracy result appeared very strong, but testing on a completely unseen YCB-Video scene revealed a substantial generalization gap.

The final unseen-scene accuracy was:

57.3%

The main conclusion is therefore not that the classifier simply “failed”, but that the evaluation exposed a scene-level generalization problem that was hidden by the original sample-level split.

This provides a useful baseline for the next stage of the project, where different classical models can be compared using the same feature representation.

Future experiments should compare models such as:

* Logistic Regression
* Decision Tree
* Random Forest
* Boosted Trees

and determine whether the observed generalization limitation is primarily related to the classifier or to the handcrafted feature representation itself.