from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from comp90051_project.airbnb_features import clean_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact text-only inputs for Colab SBERT embedding."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/inside_airbnb_australia"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("train", "test", "generalization"), required=True)
    parser.add_argument("--feature-name")
    parser.add_argument("--max-reviews-per-listing", type=int, default=50)
    parser.add_argument("--max-text-chars", type=int, default=8000)
    return parser.parse_args()


def build_snapshot_input(
    raw_dir: Path,
    city: str,
    snapshot: str,
    keys: pd.DataFrame,
    max_reviews_per_listing: int,
    max_text_chars: int,
) -> pd.DataFrame:
    listings_path = raw_dir / city / snapshot / "listings.csv.gz"
    reviews_path = raw_dir / city / snapshot / "reviews.csv.gz"
    listings = pd.read_csv(listings_path, usecols=["id", "host_about"], low_memory=False)
    listings["listing_id"] = listings["id"].astype("int64")
    listings["host_about_text"] = listings["host_about"].map(clean_text).str[:max_text_chars]

    reviews = pd.read_csv(
        reviews_path, usecols=["listing_id", "date", "comments"], low_memory=False
    )
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews[reviews["date"] < pd.Timestamp(snapshot)]
    reviews = reviews.sort_values(["listing_id", "date"], ascending=[True, False])
    reviews = reviews.groupby("listing_id", sort=False).head(max_reviews_per_listing)
    reviews["comments"] = reviews["comments"].map(clean_text)
    review_text = reviews.groupby("listing_id")["comments"].agg(" ".join).str[:max_text_chars]

    frame = keys.merge(listings[["listing_id", "host_about_text"]], how="left", on="listing_id")
    frame = frame.merge(
        review_text.rename("review_text"), how="left", left_on="listing_id", right_index=True
    )
    frame["host_about_text"] = frame["host_about_text"].fillna("")
    frame["review_text"] = frame["review_text"].fillna("")
    return frame


def main() -> None:
    args = parse_args()
    feature_name = args.feature_name or f"airbnb_features_{args.split}.parquet"
    feature_path = args.processed_dir / feature_name
    keys = pd.read_parquet(feature_path, columns=["city", "snapshot", "listing_id"])
    snapshots = keys[["city", "snapshot"]].drop_duplicates().sort_values(["city", "snapshot"])

    parts = []
    for city, snapshot in snapshots.itertuples(index=False, name=None):
        snapshot_keys = keys[keys["city"].eq(city) & keys["snapshot"].eq(snapshot)]
        if snapshot_keys.empty:
            continue
        part = build_snapshot_input(
            args.raw_dir,
            city,
            snapshot,
            snapshot_keys,
            args.max_reviews_per_listing,
            args.max_text_chars,
        )
        parts.append(part)
        print(f"Prepared {len(part):,} rows for {args.split} {city} {snapshot}", flush=True)

    if not parts:
        raise FileNotFoundError(f"No SBERT input rows built for split {args.split}")

    data = pd.concat(parts, ignore_index=True)
    output_path = args.processed_dir / f"airbnb_sbert_input_{args.split}.parquet"
    data.to_parquet(output_path, index=False)
    print(f"Wrote {len(data):,} rows and {len(data.columns):,} columns to {output_path}")


if __name__ == "__main__":
    main()
