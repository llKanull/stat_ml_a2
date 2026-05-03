from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np


ParameterValue = object


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...


EstimatorFactory = Callable[[dict[str, ParameterValue]], Estimator]
ScoreFunction = Callable[[Estimator, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class TuningResult:
    params: dict[str, ParameterValue]
    mean_score: float
    fold_scores: list[float]


def parameter_grid(grid: dict[str, Sequence[ParameterValue]]) -> list[dict[str, ParameterValue]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    if any(len(v) == 0 for v in values):
        raise ValueError("parameter grid values must be non-empty")
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    *,
    estimator_factory: EstimatorFactory,
    param_grid: dict[str, Sequence[ParameterValue]],
    inner_splits: Iterable[Split],
    score_function: ScoreFunction,
) -> tuple[dict[str, ParameterValue], list[TuningResult]]:
    X = np.asarray(X)
    y = np.asarray(y)
    results: list[TuningResult] = []
    splits = list(inner_splits)
    if not splits:
        raise ValueError("inner_splits must contain at least one split")

    for params in parameter_grid(param_grid):
        fold_scores = []
        for train_idx, val_idx in splits:
            estimator = estimator_factory(params)
            estimator.fit(X[train_idx], y[train_idx])
            score = score_function(estimator, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            fold_scores.append(float(score))
        results.append(TuningResult(params, float(np.mean(fold_scores)), fold_scores))

    best = max(results, key=lambda result: result.mean_score)
    return dict(best.params), results
