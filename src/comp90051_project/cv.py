from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def k_fold_indices(
    n_samples: int,
    k: int = 10,
    *,
    shuffle: bool = True,
    random_state: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")
    if k < 2:
        raise ValueError("k must be at least 2")
    if k > n_samples:
        raise ValueError("k cannot exceed n_samples")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        yield train_idx, test_idx
        start = stop


def repeated_train_test_split_indices(
    n_samples: int,
    *,
    n_repeats: int = 10,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    n_test = max(1, int(round(n_samples * test_size)))
    n_test = min(n_test, n_samples - 1)

    for _ in range(n_repeats):
        indices = rng.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        yield train_idx, test_idx
