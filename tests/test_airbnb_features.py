from __future__ import annotations

from pathlib import Path

import pandas as pd

from comp90051_project.airbnb_features import (
    BuildConfig,
    clean_text,
    feature_groups,
    listing_availability_targets,
    review_aggregates,
    selected_snapshots,
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
    assert pd.notna(features.loc[0, "review_sentiment_compound_mean"])
    assert features.loc[0, "review_sentiment_compound_std"] > 0


def test_listing_availability_targets_use_availability_90() -> None:
    listings = pd.DataFrame(
        {
            "id": [1, 2],
            "availability_90": [30, 120],
        }
    )

    targets = listing_availability_targets(listings, horizon_days=90)

    assert targets.loc[0, "target_available_rate_90"] == 1 / 3
    assert targets.loc[0, "target_unavailable_rate_90"] == 2 / 3
    assert targets.loc[1, "target_available_rate_90"] == 1.0


def test_default_snapshot_order_prefers_latest() -> None:
    config = BuildConfig(max_snapshots=1)

    assert config.snapshot_order == "latest"


def test_selected_snapshots_can_exclude_latest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "city": ["a", "a", "a", "b"],
            "snapshot": ["2025-01-01", "2025-04-01", "2025-07-01", "2025-03-01"],
            "file_name": ["listings.csv.gz"] * 4,
            "url": ["u"] * 4,
        }
    ).to_csv(raw_dir / "manifest.csv", index=False)

    snapshots = selected_snapshots(BuildConfig(raw_dir=raw_dir, snapshot_split="before-latest"))

    assert snapshots["snapshot"].tolist() == ["2025-01-01", "2025-04-01"]


def test_selected_snapshots_can_choose_previous_snapshot(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "city": ["a", "a", "a", "b"],
            "snapshot": ["2025-01-01", "2025-04-01", "2025-07-01", "2025-03-01"],
            "file_name": ["listings.csv.gz"] * 4,
            "url": ["u"] * 4,
        }
    ).to_csv(raw_dir / "manifest.csv", index=False)
    snapshots = selected_snapshots(BuildConfig(raw_dir=raw_dir, snapshot_split="previous"))

    assert snapshots[["city", "snapshot"]].values.tolist() == [["a", "2025-04-01"]]


def test_feature_groups_keep_targets_out_of_model_features() -> None:
    data = pd.DataFrame(
        columns=[
            "city",
            "snapshot",
            "listing_id",
            "accommodates",
            "description_sentiment_compound",
            "host_about_sentiment_compound",
            "review_sentiment_compound_mean",
            "target_unavailable_rate_90",
        ]
    )

    groups = feature_groups(data)

    assert "target_unavailable_rate_90" in groups["targets"]
    assert "target_unavailable_rate_90" not in groups["model_all"]
    assert "accommodates" in groups["base_controls"]
    assert "description_sentiment_compound" in groups["description_nlp"]
    assert "host_about_sentiment_compound" in groups["host_nlp"]
    assert "review_sentiment_compound_mean" in groups["review_nlp"]
