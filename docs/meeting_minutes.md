# Meeting Minutes

## Meeting 1

- Date: 30 April 2026
- Attendees: Anuj, Kezia
- Decisions:
  - The project will use tabular data.
  - Each team member should come to the next meeting with a research question,
    topic, and dataset.
  - Code responsibilities:
    - Cross-validation and hyperparameter tuning: Anuj
    - Feature construction and preprocessing: Kezia
  - Report responsibilities:
    - Introduction: Kezia
    - Literature review: Anuj
    - Methods, results, and discussion: Each person
    - Conclusion: Kezia
- Actions:
  - Each team member to propose a research question, topic, and dataset for the
    next meeting.
  - Allocate algorithm responsibilities for basic, intermediate, and advanced
    models.

## Meeting 2

- Date: 2 May 2026
- Attendees: Anuj, Kezia, Panda
- Decisions:
  - Candidate project topics were discussed:
    - Anuj suggested predicting stock prices using candlestick chart images.
    - Kezia suggested Airbnb listing price prediction using out-of-distribution
      cross-city generalization.
    - Panda suggested Airbnb listing price prediction incorporating additional
      features such as tourist attractions and city popularity.
- Actions:
  - Decide which candidate project topic to pursue.

## Meeting 3

- Date: 13 May 2026
- Attendees: Anuj, Kezia, Panda
- Topic suggestions:
  - Airbnb: Incorporate sentiment extracted from Airbnb host profiles to improve
    popularity prediction compared with structured listing features alone.
  - Airbnb: Use NLP-derived review signals to improve popularity prediction
    compared with structured listing features.
  - Stock: Test whether candlestick-chart image representations generated from
    OHLCV data improve short-horizon price-direction prediction robustness
    compared with numerical technical indicators.
  - Airbnb: Test whether review sentiment over time gives early signals of
    rising or declining demand, using review NLP to predict occupancy or demand.
- Decisions:
  - Prioritise the Airbnb direction because it supports structured features,
    text-derived features, temporal splits, and distribution-shift evaluation.
  - Include a data architecture discussion in the project design, especially how
    temporal splits will avoid leakage from future snapshots and reviews.
  - Compare structured listing features against NLP-enhanced feature sets.
- Work distribution:
  - Anuj: data pipeline, custom cross-validation, hyperparameter tuning,
    experiment orchestration, and reproducibility checks.
  - Kezia: feature-set updates, model/evaluation runs, generalisation analysis,
    and results tables.
  - Panda: feature engineering, literature review, alternatives considered, author/report updates,
    and final report editing.
- Actions:
  - Finalise Airbnb data architecture and temporal split design.
  - Build Airbnb feature tables with structured, host-text, and review-text
    features.
  - Record project activity logs for submission support.
