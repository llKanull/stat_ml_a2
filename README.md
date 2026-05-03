# COMP90051 Group Project

Repository for the COMP90051 2026 Semester 1 group project.

The project spec requires a machine learning research question, public dataset(s),
non-trivial feature construction/preprocessing, three distinct algorithms, custom
cross-validation, custom nested hyperparameter tuning, and custom experimental
metrics. Do not commit or submit datasets.

## Repository Layout

```text
.
├── data/                  # Local-only datasets and derived data, ignored by git
│   ├── raw/
│   └── processed/
├── docs/                  # Planning notes, meeting minutes, decision logs
├── notebooks/             # Exploratory notebooks
├── outputs/               # Generated tables/figures, ignored by git
├── report/                # Report source and bibliography
├── scripts/               # Reproducible command-line entry points
├── src/comp90051_project/ # Shared project package
└── tests/                 # Unit tests for shared code
```

## Setup

Use Python 3.11+ if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternatively, install from requirements files:

```bash
python -m pip install -r requirements-dev.txt
```

Run checks:

```bash
pytest
python scripts/smoke_check.py
```

## Project Rules To Keep Visible

- Research question must go beyond the obvious dataset task and beyond simply
  comparing three algorithms.
- Dataset must be public and cited by URL in the report.
- Tabular datasets need at least 10,000 instances and 100 original plus
  constructed features. Image datasets need at least 10,000 images and original
  images at least 100x100 pixels.
- Feature construction/preprocessing must be more substantial than one-hot
  encoding, normalization, imputation, filtering, resizing, or plain text
  processing alone.
- Compare three different algorithms with simple, medium, and complex model
  classes.
- At least one algorithm must first appear in a qualifying NeurIPS, ICML, UAI,
  AISTATS, JMLR, ICLR, or TMLR paper from 2016 onward.
- Cross-validation, nested cross-validation, hyperparameter tuning, and reported
  metrics must be implemented from scratch. Third-party plotting is allowed.
- Report at least three experimental results with error bars.
- Final Canvas submission includes a 4-page PDF report and a ZIP of code plus a
  short `Readme.txt`; do not submit data.

## Suggested Workflow

1. Fill in [docs/project_plan.md](docs/project_plan.md) before implementation.
2. Record meetings in [docs/meeting_minutes.md](docs/meeting_minutes.md).
3. Keep reusable code in `src/comp90051_project`, not only notebooks.
4. Use `scripts/run_experiment.py` as the stable entry point once the dataset and
   methods are selected.
5. Commit generated figures/tables only if they are small and needed for the
   report; raw and processed data stay local.

## Current Status

The team still needs to choose the research question, dataset(s), feature
construction, algorithms, and experiment design.
