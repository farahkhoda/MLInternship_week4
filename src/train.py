from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42


def get_base_models(random_state=RANDOM_STATE):
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'SVM': SVC(random_state=random_state, probability=True)
    }


def train_models(models, X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def tune_decision_tree(X_train, y_train, random_state=RANDOM_STATE):
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


def tune_svm(X_train, y_train, random_state=RANDOM_STATE):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(
        SVC(random_state=random_state, probability=True),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid
