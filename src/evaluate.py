import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }


def evaluate_all_models(models, X, y):
    results = {}
    for name, model in models.items():
        results[name] = compute_metrics(model, X, y)
    return pd.DataFrame(results).T


def plot_confusion_matrix(model, X_test, y_test, labels, save_path=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return cm


def plot_roc_curves(models, X_test, y_test, save_path=None):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for idx, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        axes[idx].plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        axes[idx].plot([0, 1], [0, 1], linestyle='--', color='red')
        axes[idx].set_title(name)
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curves(models, X_test, y_test, save_path=None):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for idx, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        axes[idx].plot(recall, precision, label=f'AP = {ap:.4f}')
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Recall')
        axes[idx].set_ylabel('Precision')
        axes[idx].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_misclassified(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mask = y_test.values != y_pred
    idx = np.where(mask)[0]
    return X_test.iloc[idx], y_test.iloc[idx].values, y_pred[idx]
