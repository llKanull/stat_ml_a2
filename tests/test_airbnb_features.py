from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from comp90051_project.airbnb_features import (
    BuildConfig,
    calendar_targets,
    clean_text,
    feature_groups,
    has_rows,
    review_aggregates,
)


def test_clean_text_removes_html_and_mojibake() -> None:
    text = clean_text("Regina‚Äôs <br/> place\r\nis great")

    assert text == "Regina's place is great"


def test_review_aggregates_use_only_reviews_before_cutoff() -> None:
    reviews = pd.DataFrame(
        {
            "listing_id": [1, 1, 1],
            "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "comments": ["clean and helpful", "noisy problem", "future clean review"],
        }
    )

    features = review_aggregates(reviews, max_reviews_per_listing=10, snapshot="2025-02-15")

    assert features.loc[0, "review_count_before_cutoff"] == 2
    assert features.loc[0, "review_days_since_latest"] == 14
    assert features.loc[0, "cleanliness_keyword_rate"] > 0
    assert features.loc[0, "problem_keyword_rate"] > 0


def test_calendar_targets_use_future_horizon(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv.gz"
    calendar = pd.DataFrame(
        {
            "listing_id": [1, 1, 1, 1],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-04-15"],
            "available": ["t", "f", "f", "t"],
            "price": ["$100.00", "$120.00", "$130.00", "$140.00"],
        }
    )
    calendar.to_csv(path, index=False)

    targets = calendar_targets(path, snapshot="2025-01-01", horizon_days=3)

    assert targets.loc[0, "target_calendar_days"] == 3
    assert targets.loc[0, "target_available_rate_90"] == 1 / 3
    assert targets.loc[0, "target_unavailable_rate_90"] == pytest.approx(2 / 3)


def test_empty_calendar_file_has_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty_calendar.csv.gz"
    pd.DataFrame(columns=["listing_id", "date", "available", "price", "adjusted_price"]).to_csv(
        path, index=False
    )

    assert not has_rows(path)


def test_default_snapshot_order_prefers_latest() -> None:
    config = BuildConfig(max_snapshots=1)

    assert config.snapshot_order == "latest"


def test_feature_groups_keep_targets_out_of_model_features() -> None:
    data = pd.DataFrame(
        columns=[
            "city",
            "snapshot",
            "listing_id",
            "accommodates",
            "host_about_warmth_score",
            "host_about_hash_00",
            "mean_review_sentiment",
            "reviews_hash_00",
            "target_unavailable_rate_90",
        ]
    )

    groups = feature_groups(data)

    assert "target_unavailable_rate_90" in groups["targets"]
    assert "target_unavailable_rate_90" not in groups["model_all"]
    assert "accommodates" in groups["base_controls"]
    assert "host_about_warmth_score" in groups["host_nlp"]
    assert "mean_review_sentiment" in groups["review_nlp"]
