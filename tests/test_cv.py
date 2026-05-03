import numpy as np

from comp90051_project.cv import k_fold_indices, repeated_train_test_split_indices


def test_k_fold_indices_cover_each_sample_once_as_test():
    folds = list(k_fold_indices(23, k=10, random_state=51))

    tested = np.concatenate([test_idx for _, test_idx in folds])

    assert len(folds) == 10
    assert sorted(tested.tolist()) == list(range(23))
    assert all(set(train_idx).isdisjoint(test_idx) for train_idx, test_idx in folds)


def test_repeated_train_test_split_sizes():
    splits = list(
        repeated_train_test_split_indices(50, n_repeats=10, test_size=0.2, random_state=51)
    )

    assert len(splits) == 10
    assert all(len(train_idx) == 40 for train_idx, _ in splits)
    assert all(len(test_idx) == 10 for _, test_idx in splits)
