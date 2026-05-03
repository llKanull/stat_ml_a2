from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comp90051_project.cv import k_fold_indices
from comp90051_project.metrics import accuracy_score, mean_and_error_bar


def main() -> None:
    splits = list(k_fold_indices(20, k=10, random_state=51))
    assert len(splits) == 10
    assert all(len(test_idx) == 2 for _, test_idx in splits)

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy_score(y_true, y_pred) == 0.75

    mean, error = mean_and_error_bar(np.array([0.7, 0.8, 0.9]))
    assert round(mean, 2) == 0.8
    assert error > 0

    print("Smoke check passed.")


if __name__ == "__main__":
    main()
