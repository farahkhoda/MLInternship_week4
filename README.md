ML Internship — Week 4

Overview

Week 4 focused on Supervised Machine Learning model evaluation, generalization, and error analysis.

The main goal of this week was to move beyond simply training a model and learn how to determine whether a model actually performs well, generalizes to unseen data, and where and why it makes incorrect predictions.

The practical work was carried out using two classification projects:

1. YCB-Video — Main Practical Project
2. Breast Cancer Wisconsin — Extra Practice Project

Detailed methodology, experiments, metrics, error analysis, and conclusions for each project are documented separately in the reports/ directory.

⸻

Main Quest — Supervised Learning & Model Evaluation

Learning Goals

This week focused on the following concepts:

* Train / Validation / Test split
* Data leakage prevention
* Feature scaling and StandardScaler
* Logistic Regression
* Classification evaluation
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Error Analysis
* Bias and Variance
* Overfitting and Underfitting
* Generalization
* Model performance on unseen data
* Model evaluation methodology
* Interpreting differences between validation and test performance

Advanced Learning Algorithms

The theoretical part of the week also covered:

* Model development strategies
* Bias vs. variance
* Learning curves
* Error analysis
* Model selection
* Regularization concepts
* Classical machine-learning algorithms

Neural-network sections were excluded from this week’s assigned work.

⸻

Practical Projects

1. YCB-Video — Classical Classification Evaluation

The main practical project for Week 4 uses the YCB-Video dataset.

The goal was to evaluate a classical image-based classifier under a more realistic generalization setting rather than relying only on a random sample-level train/test split.

Work Completed

* Selected and prepared a subset of the YCB-Video dataset
* Explored scenes, object classes, and class distributions
* Loaded RGB images and visible-object masks
* Extracted numerical image features from masked object regions
* Built a feature dataset for classical machine learning
* Created training, validation, and test sets
* Applied feature scaling using training data only
* Trained a Logistic Regression classifier
* Generated predictions for Train, Validation, and Test sets
* Evaluated the classifier using:
    * Accuracy
    * Precision
    * Recall
    * F1-score
    * Confusion Matrix
* Performed error analysis
* Examined class-level prediction failures
* Tested generalization to a completely unseen scene
* Analyzed the relationship between validation performance and unseen-scene performance
* Investigated possible bias/variance and generalization issues

Documentation

The complete technical report is available here:

reports/REPORT.md

Notebook:

notebooks/ycbv.ipynb

⸻

2. Breast Cancer Wisconsin — Extra Practice

The Breast Cancer Wisconsin classification project was completed as an additional practice project for the concepts studied during Week 4.

The purpose of this project was to reinforce the model evaluation workflow on a clean numerical classification dataset.

Work Completed

* Dataset exploration and data-quality checks
* Train / Validation / Test splitting
* Feature scaling
* Logistic Regression
* Decision Tree
* Support Vector Machine
* Classification metrics
* Confusion matrices
* ROC / AUC analysis
* Cross-validation
* Learning curves
* Hyperparameter search
* Error analysis
* Bias / variance analysis
* Model comparison and interpretation

This project served as additional practice for applying the evaluation concepts learned during the week.

Its complete methodology and results are documented separately in the project report.

⸻

Key Skills Practiced

By the end of Week 4, I practiced how to:

* Build a reproducible supervised-learning workflow
* Separate training, validation, and test data correctly
* Prevent data leakage during preprocessing
* Evaluate classification models using multiple metrics
* Interpret confusion matrices
* Investigate individual model errors
* Distinguish validation performance from true generalization
* Evaluate performance on unseen data distributions
* Identify possible overfitting and underfitting patterns
* Analyze bias and variance
* Compare model behavior rather than relying only on accuracy
* Document machine-learning experiments in a reproducible way

⸻

Project Structure

MLInternship_week4/
│
├── notebooks/
│   └── ycbv.ipynb
│
├── reports/
│   └── REPORT.md
│
├── README.md
├── .gitignore
│
└── test/                  # Local YCB-Video dataset — not tracked by Git

The datasets are kept outside Git tracking because of their size.
The notebooks and reports contain the methodology required to reproduce and understand the experiments.

⸻

Week 4 Outcome

Week 4 shifted the focus from simply training a model to understanding how trustworthy its performance is.

The main practical lesson was that strong performance on a validation set does not necessarily guarantee strong performance on a genuinely unseen environment or distribution.

The YCB-Video project provided practical experience with this distinction, while the Breast Cancer project provided additional practice with standard classification evaluation and model analysis.

⸻

Next Step — Week 5

The next stage will focus on Advanced Learning Algorithms and classical model comparison, including:

* Decision Trees
* Random Forest
* Ensemble methods
* Boosted Trees
* XGBoost concepts

The practical work will compare multiple classical models on the same feature representation and examine their strengths, weaknesses, and behavior.

