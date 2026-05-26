COMP90051 Project Code

Topic: predicting high-demand Airbnb listings using structured listing features
and NLP-derived host/review text features.

Setup:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev]"

Checks:
  pytest
  python scripts/smoke_check.py

Run:
  bash scripts/run_pipeline.sh

Fast run:
  bash scripts/run_pipeline.sh --fast

Data is downloaded from Inside Airbnb by the pipeline. Raw data is not included
in the submission.
