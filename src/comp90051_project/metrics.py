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
    predicted_positive = matrix.sum(axis=0).astype(float)
    actual_positive = matrix.sum(axis=1).astype(float)

    precision = _safe_divide(tp, predicted_positive)
    recall = _safe_divide(tp, actual_positive)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }

def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    _validate_same_shape(y_true, scores)

    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "roc_auc_score requires at least one positive and one negative "
            f"sample.  Got n_pos={n_pos}, n_neg={n_neg}."
        )

    # Sort by score ascending and assign 1-indexed ranks with tie averaging.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        # Find the end of the current tied group.
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # average of 1-indexed positions i+1..j
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def mean_and_error_bar(
    values: np.ndarray, confidence_z: float = 1.96
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if len(values) == 1:
        return float(values[0]), 0.0

    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, confidence_z * standard_error

def _safe_divide(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Arrays must have the same shape, got {a.shape} and {b.shape}."
        )