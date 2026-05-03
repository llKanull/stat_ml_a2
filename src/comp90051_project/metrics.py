from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _validate_same_shape(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | list | None = None,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _validate_same_shape(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    labels = np.asarray(labels)
    label_to_pos = {label: pos for pos, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[label_to_pos[actual], label_to_pos[predicted]] += 1
    return matrix


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | list | None = None,
) -> dict[str, np.ndarray | float]:
    matrix = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)

    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, actual)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def mean_and_error_bar(values: np.ndarray, confidence_z: float = 1.96) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if len(values) == 1:
        return float(values[0]), 0.0

    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, confidence_z * standard_error


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def _validate_same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
