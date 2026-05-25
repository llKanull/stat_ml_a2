from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from comp90051_project.airbnb_features import feature_groups


DEFAULT_GENERALIZATION_CITIES = ("sydney", "tasmania", "western-australia")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train, temporal-test, and held-out-city Airbnb evaluation parquets."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--previous-name", default="airbnb_features_previous_all.parquet")
    parser.add_argument("--latest-name", default="airbnb_features_latest.parquet")
    parser.add_argument("--train-name", default="airbnb_features_train.parquet")
    parser.add_argument("--test-name", default="airbnb_features_test.parquet")
    parser.add_argument("--generalization-name", default="airbnb_features_generalization.parquet")
    parser.add_argument(
        "--generalization-city",
        action="append",
        dest="generalization_cities",
        default=None,
        help="City to withhold for held-out-city generalization. Can be repeated.",
    )
    parser.add_argument("--min-reviews-ltm-for-popular", type=int, default=1)
    return parser.parse_args()


def assign_labels(
    frame: pd.DataFrame,
    *,
    demand_threshold: float,
    popularity_threshold: float,
) -> pd.DataFrame:
    return frame.drop(
        columns=["target_high_demand", "target_high_popularity"],
        errors="ignore",
    ).assign(
        target_high_demand=lambda df: (df["target_unavailable_rate_90"] >= demand_threshold).astype(
            int
        ),
        target_high_popularity=lambda df: (df["target_reviews_ltm"] >= popularity_threshold).astype(
            int
        ),
    )


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def filter_active_short_term(frame: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)

    if "minimum_nights" in frame.columns:
        mask &= frame["minimum_nights"] <= 90
    if "has_availability" in frame.columns:
        mask &= frame["has_availability"].isin([True, "t", 1])
    if "number_of_reviews" in frame.columns:
        mask &= frame["number_of_reviews"] > 0

    missing = [
        c
        for c in ["minimum_nights", "has_availability", "number_of_reviews"]
        if c not in frame.columns
    ]
    if missing:
        print(f"filter_active_short_term: columns not found, skipped: {missing}")

    return frame[mask].copy()


def main() -> None:
    args = parse_args()
    heldout = set(args.generalization_cities or DEFAULT_GENERALIZATION_CITIES)

    previous = filter_active_short_term(pd.read_parquet(args.processed_dir / args.previous_name))
    latest = filter_active_short_term(pd.read_parquet(args.processed_dir / args.latest_name))

    train = previous[~previous["city"].isin(heldout)].copy()
    test = latest[latest["city"].isin(train["city"].unique())].copy()
    generalization = latest[latest["city"].isin(heldout)].copy()

    if train.empty:
        raise ValueError("Train split is empty.")
    if test.empty:
        raise ValueError("Temporal test split is empty.")
    if generalization.empty:
        raise ValueError("Generalization split is empty.")

    demand_threshold = float(train["target_unavailable_rate_90"].quantile(0.75))
    popularity_threshold = float(
        max(args.min_reviews_ltm_for_popular, train["target_reviews_ltm"].quantile(0.75))
    )
    train = assign_labels(
        train,
        demand_threshold=demand_threshold,
        popularity_threshold=popularity_threshold,
    )
    test = assign_labels(
        test,
        demand_threshold=demand_threshold,
        popularity_threshold=popularity_threshold,
    )
    generalization = assign_labels(
        generalization,
        demand_threshold=demand_threshold,
        popularity_threshold=popularity_threshold,
    )

    write_parquet(train, args.processed_dir / args.train_name)
    write_parquet(test, args.processed_dir / args.test_name)
    write_parquet(generalization, args.processed_dir / args.generalization_name)

    groups_path = args.processed_dir / "airbnb_feature_groups.json"
    groups_path.write_text(json.dumps(feature_groups(train), indent=2) + "\n")

    print(f"Thresholds: demand={demand_threshold:.6f}, popularity={popularity_threshold:.6f}")
    for name, frame in [
        ("train", train),
        ("test", test),
        ("generalization", generalization),
    ]:
        print(
            f"{name}: {len(frame):,} rows, {frame['city'].nunique()} cities, "
            f"high_demand={frame['target_high_demand'].mean():.3f}"
        )
        print("  " + ", ".join(sorted(frame["city"].unique())))


if __name__ == "__main__":
    main()
