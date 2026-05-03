from comp90051_project.tuning import parameter_grid


def test_parameter_grid_expands_combinations():
    grid = parameter_grid({"alpha": [0.1, 1.0], "depth": [2, 3]})

    assert grid == [
        {"alpha": 0.1, "depth": 2},
        {"alpha": 0.1, "depth": 3},
        {"alpha": 1.0, "depth": 2},
        {"alpha": 1.0, "depth": 3},
    ]
