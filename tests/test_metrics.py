import numpy as np

from comp90051_project.metrics import accuracy_score, confusion_matrix, mean_and_error_bar


def test_accuracy_score():
    assert accuracy_score(np.array([1, 0, 1]), np.array([1, 1, 1])) == 2 / 3


def test_confusion_matrix():
    matrix = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), labels=[0, 1])

    np.testing.assert_array_equal(matrix, np.array([[1, 1], [0, 2]]))


def test_mean_and_error_bar_single_value():
    assert mean_and_error_bar(np.array([0.5])) == (0.5, 0.0)
