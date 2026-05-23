from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from comp90051_project.airbnb_features import (
    BuildConfig,
    build_dataset,
    build_snapshot_features,
    feature_groups,
    has_rows,
    selected_snapshots,
    write_dataset,
    write_feature_groups,
)


def build_chunk(task: tuple[BuildConfig, str, str]) -> tuple[Path | None, int]:
    config, city, snapshot = task
    listings_path = config.raw_dir / city / snapshot / "listings.csv.gz"
    reviews_path = config.raw_dir / city / snapshot / "reviews.csv.gz"
    calendar_path = config.raw_dir / city / snapshot / "calendar.csv.gz"
    if not (listings_path.exists() and reviews_path.exists() and calendar_path.exists()):
        return None, 0
    if not has_rows(calendar_path):
        return None, 0
    part = build_snapshot_features(
        listings_path, reviews_path, calendar_path, city, snapshot, config
    )
    part_path = write_dataset(
        part,
        config.output_dir / "snapshots",
        f"airbnb_features_{city}_{snapshot}.parquet",
    )
    return part_path, len(part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned Airbnb features for modelling.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/inside_airbnb_australia"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-name", default="airbnb_features.parquet")
    parser.add_argument("--max-review-rows", type=int)
    parser.add_argument("--max-reviews-per-listing", type=int, default=50)
    parser.add_argument("--target-horizon-days", type=int, default=90)
    parser.add_argument("--city", action="append", dest="cities")
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--snapshot-order", choices=("latest", "earliest"), default="latest")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_review_rows=args.max_review_rows,
        max_reviews_per_listing=args.max_reviews_per_listing,
        target_horizon_days=args.target_horizon_days,
        cities=tuple(args.cities) if args.cities else None,
        max_snapshots=args.max_snapshots,
        snapshot_order=args.snapshot_order,
    )
    if args.chunked:
        tasks = [
            (config, city, snapshot)
            for city, snapshot in selected_snapshots(config)[["city", "snapshot"]].itertuples(
                index=False,
                name=None,
            )
        ]
        chunk_paths: list[Path] = []
        if args.jobs == 1:
            for task in tasks:
                path, rows = build_chunk(task)
                if rows and path is not None:
                    print(f"Wrote chunk {rows:,} rows to {path}", flush=True)
                    chunk_paths.append(path)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = [executor.submit(build_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    path, rows = future.result()
                    if rows and path is not None:
                        print(f"Wrote chunk {rows:,} rows to {path}", flush=True)
                        chunk_paths.append(path)
        if not chunk_paths:
            raise FileNotFoundError("No snapshot chunks were built.")
        parts = [pd.read_parquet(path) for path in sorted(chunk_paths)]
        data = pd.concat(parts, ignore_index=True, sort=False)
        data["target_high_demand"] = (
            data["target_unavailable_rate_90"] >= data["target_unavailable_rate_90"].quantile(0.75)
        ).astype(int)
        data["target_high_popularity"] = (
            data["target_reviews_ltm"] >= max(1, data["target_reviews_ltm"].quantile(0.75))
        ).astype(int)
    else:
        data = build_dataset(config)
    path = write_dataset(data, args.output_dir, args.output_name)
    groups_path = write_feature_groups(data, args.output_dir)
    groups = feature_groups(data)
    print(f"Wrote {len(data):,} rows and {len(data.columns):,} columns to {path}")
    print(f"Wrote feature groups to {groups_path}")
    print(
        "Feature groups: "
        f"base={len(groups['base_controls'])}, "
        f"host={len(groups['host_all'])}, "
        f"review={len(groups['review_all'])}, "
        f"targets={len(groups['targets'])}"
    )
    print(f"High-demand class rate: {data['target_high_demand'].mean():.3f}")
    print(f"High-popularity class rate: {data['target_high_popularity'].mean():.3f}")


if __name__ == "__main__":
    main()
