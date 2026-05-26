# COMP90051 Airbnb Demand Prediction

This repository contains the code and report for the COMP90051 2026 Semester 1
group project. The final question is:

> Do NLP-derived host profile and review features improve prediction of
> high-demand Airbnb listings beyond structured listing features alone?

The project uses public Inside Airbnb Australia snapshots, engineered structured
and text features, custom cross-validation and tuning code, and three model
classes: Logistic Regression, CatBoost, and FT-Transformer.

## Layout

```text
data/                  Local raw and processed data; not submitted
docs/                  Meeting minutes, planning notes, submission logs
outputs/               Generated experiment tables
report/                Final PDF and LaTeX source
scripts/               Pipeline and experiment entry points
src/comp90051_project/ Shared feature, metric, CV, tuning, and model code
tests/                 Unit tests for shared code
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproduce

Run checks:

```bash
pytest
python scripts/smoke_check.py
```

Run the full pipeline:

```bash
bash scripts/run_pipeline.sh
```

Useful shorter runs:

```bash
bash scripts/run_pipeline.sh --fast
bash scripts/run_pipeline.sh --skip-download --skip-fttransformer
```

The pipeline downloads Inside Airbnb data, builds feature parquets, creates
train/test/geographic-generalisation splits, runs nested-CV experiments, and
writes result tables to `outputs/tables/`.

## Submission Notes

- Submit the report PDF and code ZIP only.
- Do not submit raw data.
- `Readme.txt` is the short submission README.
- `docs/meeting_minutes.md` and `docs/git_log.md` are included as supporting
  project records.
