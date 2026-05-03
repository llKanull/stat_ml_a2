How to run the project code:

1. Create and activate a Python environment.
2. Install dependencies with:
   python -m pip install -e ".[dev]"
3. Run checks with:
   pytest
   python scripts/smoke_check.py
4. Run the final experiment with:
   python scripts/run_experiment.py

Note: datasets are not included in this repository and should not be submitted.
