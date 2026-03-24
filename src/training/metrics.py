from __future__ import annotations

import math

import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
except Exception:  # pragma: no cover
    accuracy_score = None
    f1_score = None
    mean_absolute_error = None
    mean_squared_error = None
    precision_score = None
    r2_score = None
    recall_score = None

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if mean_squared_error is not None:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    else:
        residual = y_true - y_pred
        mse = float(np.mean(np.square(residual)))
        mae = float(np.mean(np.abs(residual)))
        denom = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = 1.0 - float(np.sum(np.square(residual))) / denom if denom > 0 else 0.0
    metrics = {
        "mae": float(mae),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2),
    }
    if spearmanr is not None:
        corr, _ = spearmanr(y_true, y_pred)
        metrics["spearman"] = float(corr) if corr == corr else 0.0
    return metrics


def multioutput_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, float]:
    results = {}
    for index, name in enumerate(target_names):
        target_metrics = regression_metrics(y_true[:, index], y_pred[:, index])
        for metric_name, value in target_metrics.items():
            results[f"{name}_{metric_name}"] = value
    return results


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if accuracy_score is None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        accuracy = float(np.mean(y_true == y_pred))
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        precisions = []
        recalls = []
        f1s = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return {
            "accuracy": accuracy,
            "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
            "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
            "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
