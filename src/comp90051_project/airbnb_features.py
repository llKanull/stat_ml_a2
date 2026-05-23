from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


TEXT_HASH_FEATURES = 64
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")
TAG_RE = re.compile(r"<[^>]+>")
MONEY_RE = re.compile(r"[^0-9.]")
MOJIBAKE_REPLACEMENTS = {
    "‚Äô": "'",
    "‚Äò": "'",
    "‚Äú": '"',
    "‚Äù": '"',
    "‚Äì": "-",
    "‚Äî": "-",
    "¬†": " ",
}

POSITIVE_WORDS = frozenset(
    {
        "amazing",
        "beautiful",
        "best",
        "clean",
        "comfortable",
        "convenient",
        "easy",
        "excellent",
        "fantastic",
        "friendly",
        "good",
        "great",
        "happy",
        "helpful",
        "highly",
        "love",
        "lovely",
        "perfect",
        "recommend",
        "relaxed",
        "sunny",
        "valuable",
        "welcome",
        "wonderful",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "broken",
        "cold",
        "dirty",
        "difficult",
        "disappointing",
        "issue",
        "late",
        "loud",
        "missing",
        "noisy",
        "old",
        "poor",
        "problem",
        "small",
        "uncomfortable",
        "worse",
    }
)
HOST_LEXICONS = {
    "warmth": frozenset(
        {"welcome", "happy", "friendly", "love", "enjoy", "company", "home", "meet", "share"}
    ),
    "professionalism": frozenset(
        {"hosting", "co-hosting", "service", "ambassador", "managed", "team", "reviews"}
    ),
    "local_expertise": frozenset(
        {"melbourne", "local", "suburb", "restaurant", "guide", "recommendations", "beach", "city"}
    ),
    "travel_culture": frozenset(
        {"travel", "traveller", "travelled", "journey", "cultures", "international", "countries"}
    ),
    "arts_lifestyle": frozenset(
        {"artist", "animation", "video", "arts", "music", "cinema", "screenwriter", "decorator"}
    ),
    "family_home": frozenset({"family", "wife", "grandfather", "children", "boys", "girls", "dog"}),
}
REVIEW_LEXICONS = {
    "cleanliness": frozenset({"clean", "spotless", "tidy", "dirty", "dusty"}),
    "location": frozenset({"location", "tram", "train", "central", "walk", "cbd", "beach"}),
    "host": frozenset({"host", "friendly", "helpful", "responsive", "communicative", "welcoming"}),
    "comfort": frozenset({"comfortable", "bed", "quiet", "room", "spacious", "small", "noisy"}),
    "value": frozenset({"value", "price", "expensive", "cheap", "worth"}),
    "noise": frozenset({"noise", "noisy", "loud", "quiet", "street"}),
    "checkin": frozenset({"checkin", "check-in", "arrival", "key", "access"}),
    "amenity": frozenset({"wifi", "kitchen", "shower", "parking", "washer", "aircon", "heating"}),
    "problem": frozenset({"problem", "issue", "broken", "missing", "late", "difficult"}),
}


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path = Path("data/raw/inside_airbnb_australia")
    output_dir: Path = Path("data/processed")
    max_review_rows: int | None = None
    max_reviews_per_listing: int = 50
    target_horizon_days: int = 90
    min_reviews_ltm_for_popular: int = 1
    cities: tuple[str, ...] | None = None
    max_snapshots: int | None = None
    snapshot_order: str = "latest"


def load_manifest(raw_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(raw_dir / "manifest.csv")
    return manifest[
        manifest["file_name"].isin({"listings.csv.gz", "reviews.csv.gz", "calendar.csv.gz"})
    ].copy()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value))
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = text.replace("\r", " ").replace("\n", " ")
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def lexical_features(
    texts: pd.Series, prefix: str, lexicons: dict[str, frozenset[str]]
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for text in texts.fillna("").map(clean_text):
        tokens = tokenise(text)
        token_count = len(tokens)
        unique_count = len(set(tokens))
        sentence_count = max(1, len([part for part in SENTENCE_RE.split(text) if part.strip()]))
        positive_count = sum(token in POSITIVE_WORDS for token in tokens)
        negative_count = sum(token in NEGATIVE_WORDS for token in tokens)
        first_person = sum(token in {"i", "me", "my", "we", "our", "us"} for token in tokens)
        second_person = sum(token in {"you", "your", "guests", "guest"} for token in tokens)

        row = {
            f"{prefix}_present": float(bool(text)),
            f"{prefix}_char_count": float(len(text)),
            f"{prefix}_word_count": float(token_count),
            f"{prefix}_sentence_count": float(sentence_count),
            f"{prefix}_avg_sentence_words": token_count / sentence_count if token_count else 0.0,
            f"{prefix}_lexical_diversity": unique_count / token_count if token_count else 0.0,
            f"{prefix}_positive_rate": positive_count / token_count if token_count else 0.0,
            f"{prefix}_negative_rate": negative_count / token_count if token_count else 0.0,
            f"{prefix}_sentiment_balance": (positive_count - negative_count) / token_count
            if token_count
            else 0.0,
            f"{prefix}_first_person_rate": first_person / token_count if token_count else 0.0,
            f"{prefix}_second_person_rate": second_person / token_count if token_count else 0.0,
            f"{prefix}_exclamation_count": float(text.count("!")),
            f"{prefix}_question_count": float(text.count("?")),
            f"{prefix}_contains_url": float("http" in text.lower() or ".com" in text.lower()),
            f"{prefix}_contains_year": float(bool(re.search(r"\b(?:19|20)\d{2}\b", text))),
        }
        token_set = set(tokens)
        for name, words in lexicons.items():
            row[f"{prefix}_{name}_score"] = float(len(token_set & words))
            row[f"{prefix}_{name}_rate"] = (
                sum(token in words for token in tokens) / token_count if token_count else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows, index=texts.index)


def hashed_text_features(
    texts: pd.Series, prefix: str, n_features: int = TEXT_HASH_FEATURES
) -> pd.DataFrame:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    matrix = vectorizer.transform(texts.fillna("").map(clean_text))
    columns = [f"{prefix}_hash_{index:02d}" for index in range(n_features)]
    return pd.DataFrame(matrix.toarray(), columns=columns, index=texts.index)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def percentage_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column].astype(str).str.rstrip("%"), errors="coerce") / 100.0


def money_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index)
    cleaned = df[column].map(lambda value: MONEY_RE.sub("", str(value)))
    return pd.to_numeric(cleaned, errors="coerce")


def boolean_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index)
    return df[column].map({"t": 1.0, "f": 0.0, True: 1.0, False: 0.0})


def amenity_counts(values: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    groups = {
        "work": {"wifi", "workspace", "ethernet", "desk"},
        "family": {"crib", "high chair", "children", "baby", "washer", "kitchen"},
        "luxury": {"pool", "hot tub", "sauna", "gym", "waterfront", "bbq"},
        "safety": {"smoke alarm", "carbon monoxide", "fire extinguisher", "first aid"},
    }
    for value in values.fillna(""):
        try:
            amenities = json.loads(value) if isinstance(value, str) else []
        except json.JSONDecodeError:
            amenities = []
        text = " ".join(str(item).lower() for item in amenities)
        row = {"amenity_count": float(len(amenities))}
        for name, words in groups.items():
            row[f"amenity_{name}_score"] = float(sum(word in text for word in words))
        rows.append(row)
    return pd.DataFrame(rows, index=values.index)


def missingness_indicators(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{column}_missing": frame[column].isna().astype(float)
            for column in columns
            if column in frame
        },
        index=frame.index,
    )


def city_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    lat = numeric_series(df, "latitude")
    lon = numeric_series(df, "longitude")
    city_lat = lat.median()
    city_lon = lon.median()
    km_per_degree_lat = 111.32
    km_per_degree_lon = 111.32 * np.cos(np.deg2rad(city_lat))
    distance = np.sqrt(
        ((lat - city_lat) * km_per_degree_lat) ** 2 + ((lon - city_lon) * km_per_degree_lon) ** 2
    )
    return pd.DataFrame({"distance_to_city_median_km": distance}, index=df.index)


def listing_features(listings: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=listings.index)
    out["listing_id"] = listings["id"].astype("int64")
    out["city"] = listings["city"]
    out["snapshot"] = listings["snapshot"]
    out["target_reviews_ltm"] = numeric_series(listings, "number_of_reviews_ltm").fillna(0.0)
    out["target_reviews_per_month"] = numeric_series(listings, "reviews_per_month").fillna(0.0)

    for column in [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "minimum_nights",
        "maximum_nights",
        "number_of_reviews",
        "review_scores_rating",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
    ]:
        out[column] = numeric_series(listings, column)

    out["price"] = money_series(listings, "price")
    out["host_response_rate"] = percentage_series(listings, "host_response_rate")
    out["host_acceptance_rate"] = percentage_series(listings, "host_acceptance_rate")
    out["host_is_superhost"] = boolean_series(listings, "host_is_superhost")
    out["host_has_profile_pic"] = boolean_series(listings, "host_has_profile_pic")
    out["host_identity_verified"] = boolean_series(listings, "host_identity_verified")
    out["instant_bookable"] = boolean_series(listings, "instant_bookable")
    out["log_price"] = np.log1p(out["price"])
    out["log_host_listing_count"] = np.log1p(numeric_series(listings, "host_total_listings_count"))
    out["log_reviews_before_cutoff"] = np.log1p(out["number_of_reviews"])
    out["price_per_guest"] = out["price"] / out["accommodates"].replace(0, np.nan)
    out["beds_per_guest"] = out["beds"] / out["accommodates"].replace(0, np.nan)
    out["bathrooms_per_guest"] = out["bathrooms"] / out["accommodates"].replace(0, np.nan)
    out["bedrooms_per_guest"] = out["bedrooms"] / out["accommodates"].replace(0, np.nan)
    out["minimum_nights_log"] = np.log1p(out["minimum_nights"])
    out["host_response_strength"] = out["host_response_rate"] * out["host_acceptance_rate"]
    out["professional_host_flag"] = (
        numeric_series(listings, "host_total_listings_count").fillna(0) > 1
    ).astype(float)
    out["shared_room_flag"] = (listings.get("room_type", "") == "Shared room").astype(float)
    out["entire_home_flag"] = (listings.get("room_type", "") == "Entire home/apt").astype(float)
    out["large_capacity_flag"] = (out["accommodates"].fillna(0) >= 6).astype(float)
    out["listing_age_days"] = (
        pd.to_datetime(listings["snapshot"], errors="coerce")
        - pd.to_datetime(listings.get("host_since"), errors="coerce")
    ).dt.days
    out["listing_age_years"] = out["listing_age_days"] / 365.25
    out["days_since_first_review"] = (
        pd.to_datetime(listings["snapshot"], errors="coerce")
        - pd.to_datetime(listings.get("first_review"), errors="coerce")
    ).dt.days
    out["days_since_last_review_listing"] = (
        pd.to_datetime(listings["snapshot"], errors="coerce")
        - pd.to_datetime(listings.get("last_review"), errors="coerce")
    ).dt.days

    categorical = ["room_type", "property_type", "neighbourhood_cleansed", "host_response_time"]
    for column in categorical:
        if column in listings:
            encoded = pd.get_dummies(listings[column].fillna("missing"), prefix=column, dtype=float)
            out = pd.concat([out, encoded], axis=1)

    out = pd.concat(
        [out, amenity_counts(listings.get("amenities", pd.Series("", index=listings.index)))],
        axis=1,
    )
    out = pd.concat([out, city_distance_features(listings)], axis=1)
    out = pd.concat(
        [
            out,
            missingness_indicators(
                out,
                [
                    "price",
                    "host_response_rate",
                    "host_acceptance_rate",
                    "review_scores_rating",
                    "bedrooms",
                    "bathrooms",
                    "beds",
                ],
            ),
        ],
        axis=1,
    )
    return out


def host_about_features(listings: pd.DataFrame) -> pd.DataFrame:
    texts = listings.get("host_about", pd.Series("", index=listings.index))
    features = lexical_features(texts, "host_about", HOST_LEXICONS)
    hashes = hashed_text_features(texts, "host_about")
    return pd.concat([features, hashes], axis=1)


def review_text_scores(text: str) -> dict[str, float]:
    tokens = tokenise(text)
    token_count = len(tokens)
    positive_count = sum(token in POSITIVE_WORDS for token in tokens)
    negative_count = sum(token in NEGATIVE_WORDS for token in tokens)
    scores = {
        "review_word_count": float(token_count),
        "review_sentiment": (positive_count - negative_count) / token_count if token_count else 0.0,
        "review_is_positive": float(positive_count > negative_count),
        "review_is_negative": float(negative_count > positive_count),
    }
    for name, words in REVIEW_LEXICONS.items():
        scores[f"review_{name}_keyword_rate"] = (
            sum(token in words for token in tokens) / token_count if token_count else 0.0
        )
    return scores


def review_aggregates(
    reviews: pd.DataFrame,
    max_reviews_per_listing: int,
    snapshot: str,
) -> pd.DataFrame:
    reviews = reviews[["listing_id", "date", "comments"]].copy()
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews[reviews["date"] < pd.Timestamp(snapshot)]
    reviews = reviews.sort_values(["listing_id", "date"], ascending=[True, False])
    reviews = reviews.groupby("listing_id", sort=False).head(max_reviews_per_listing)
    reviews["clean_comments"] = reviews["comments"].map(clean_text)
    if reviews.empty:
        return pd.DataFrame(columns=["listing_id"])

    review_scores = pd.DataFrame(
        [review_text_scores(text) for text in reviews["clean_comments"]],
        index=reviews.index,
    )
    reviews = pd.concat([reviews, review_scores], axis=1)

    grouped_text = reviews.groupby("listing_id")["clean_comments"].agg(" ".join)
    lexical = lexical_features(grouped_text, "reviews", REVIEW_LEXICONS)
    hashes = hashed_text_features(grouped_text, "reviews")

    agg_spec = {
        "review_text_count": ("clean_comments", "size"),
        "mean_review_sentiment": ("review_sentiment", "mean"),
        "sentiment_std": ("review_sentiment", "std"),
        "negative_review_share": ("review_is_negative", "mean"),
        "positive_review_share": ("review_is_positive", "mean"),
        "average_review_length": ("review_word_count", "mean"),
    }
    for name in REVIEW_LEXICONS:
        agg_spec[f"{name}_keyword_rate"] = (f"review_{name}_keyword_rate", "mean")
    counts = reviews.groupby("listing_id").agg(**agg_spec)
    counts = counts.rename(columns={"review_text_count": "review_count_before_cutoff"})
    latest = reviews.groupby("listing_id")["date"].max()
    earliest = reviews.groupby("listing_id")["date"].min()
    counts["review_days_since_latest"] = (pd.Timestamp(snapshot) - latest).dt.days
    counts["review_span_days"] = (latest - earliest).dt.days
    counts["sentiment_std"] = counts["sentiment_std"].fillna(0.0)

    return pd.concat([counts, lexical, hashes], axis=1).reset_index()


def read_listings(path: Path, city: str, snapshot: str) -> pd.DataFrame:
    listings = pd.read_csv(path, low_memory=False)
    listings["city"] = city
    listings["snapshot"] = snapshot
    return listings


def read_reviews(path: Path, max_rows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def has_rows(path: Path) -> bool:
    try:
        data = pd.read_csv(path, nrows=1)
    except pd.errors.EmptyDataError:
        return False
    return not data.empty


def calendar_targets(path: Path, snapshot: str, horizon_days: int) -> pd.DataFrame:
    calendar = pd.read_csv(path, low_memory=False)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    cutoff = pd.Timestamp(snapshot)
    end = cutoff + pd.Timedelta(days=horizon_days)
    calendar = calendar[(calendar["date"] >= cutoff) & (calendar["date"] < end)].copy()
    if calendar.empty:
        return pd.DataFrame(columns=["listing_id"])
    calendar["available_flag"] = calendar["available"].map({"t": 1.0, "f": 0.0})
    calendar["calendar_price"] = money_series(calendar, "price")
    calendar["is_weekend"] = calendar["date"].dt.dayofweek.isin([4, 5]).astype(float)

    targets = calendar.groupby("listing_id").agg(
        target_calendar_days=("available_flag", "size"),
        target_available_rate_90=("available_flag", "mean"),
        target_calendar_price_mean=("calendar_price", "mean"),
        target_calendar_price_std=("calendar_price", "std"),
    )
    weekend = (
        calendar[calendar["is_weekend"].eq(1.0)].groupby("listing_id")["calendar_price"].mean()
    )
    weekday = (
        calendar[calendar["is_weekend"].eq(0.0)].groupby("listing_id")["calendar_price"].mean()
    )
    targets["target_weekend_price_premium"] = weekend - weekday
    targets["target_unavailable_rate_90"] = 1.0 - targets["target_available_rate_90"]
    targets["target_calendar_price_std"] = targets["target_calendar_price_std"].fillna(0.0)
    targets["target_weekend_price_premium"] = targets["target_weekend_price_premium"].fillna(0.0)
    return targets.reset_index()


def build_snapshot_features(
    listings_path: Path,
    reviews_path: Path,
    calendar_path: Path,
    city: str,
    snapshot: str,
    config: BuildConfig,
) -> pd.DataFrame:
    listings = read_listings(listings_path, city, snapshot)
    listing_df = listing_features(listings)
    host_df = host_about_features(listings)
    reviews = read_reviews(reviews_path, config.max_review_rows)
    review_df = review_aggregates(reviews, config.max_reviews_per_listing, snapshot)
    target_df = calendar_targets(calendar_path, snapshot, config.target_horizon_days)

    features = pd.concat([listing_df, host_df], axis=1)
    features = features.merge(review_df, how="left", left_on="listing_id", right_on="listing_id")
    return features.merge(target_df, how="inner", left_on="listing_id", right_on="listing_id")


def selected_snapshots(config: BuildConfig) -> pd.DataFrame:
    manifest = load_manifest(config.raw_dir)
    grouped = manifest.pivot_table(
        index=["city", "snapshot"],
        columns="file_name",
        values="url",
        aggfunc="first",
    ).reset_index()
    if config.cities is not None:
        grouped = grouped[grouped["city"].isin(config.cities)]
    if config.max_snapshots is not None:
        ascending = config.snapshot_order == "earliest"
        grouped = (
            grouped.sort_values(["city", "snapshot"], ascending=[True, ascending])
            .groupby("city")
            .head(config.max_snapshots)
        )
    return grouped.sort_values(["city", "snapshot"]).reset_index(drop=True)


def build_dataset(config: BuildConfig) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    grouped = selected_snapshots(config)

    for city, snapshot in grouped[["city", "snapshot"]].itertuples(index=False, name=None):
        listings_path = config.raw_dir / city / snapshot / "listings.csv.gz"
        reviews_path = config.raw_dir / city / snapshot / "reviews.csv.gz"
        calendar_path = config.raw_dir / city / snapshot / "calendar.csv.gz"
        if listings_path.exists() and reviews_path.exists() and calendar_path.exists():
            if not has_rows(calendar_path):
                continue
            records.append(
                build_snapshot_features(
                    listings_path, reviews_path, calendar_path, city, snapshot, config
                )
            )

    if not records:
        raise FileNotFoundError(f"No listing/review snapshots found under {config.raw_dir}")

    data = pd.concat(records, ignore_index=True, sort=False)
    data["target_high_demand"] = (
        data["target_unavailable_rate_90"] >= data["target_unavailable_rate_90"].quantile(0.75)
    ).astype(int)
    data["target_high_popularity"] = (
        data["target_reviews_ltm"]
        >= max(config.min_reviews_ltm_for_popular, data["target_reviews_ltm"].quantile(0.75))
    ).astype(int)
    return data.replace([np.inf, -np.inf], np.nan)


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(data.columns)
    keys = ["city", "snapshot", "listing_id"]
    targets = [column for column in columns if column.startswith("target_")]
    host_nlp = [
        column for column in columns if column.startswith("host_about_") and "_hash_" not in column
    ]
    host_hash = [column for column in columns if column.startswith("host_about_hash_")]
    review_nlp_prefixes = (
        "review_count_before_cutoff",
        "mean_review_sentiment",
        "sentiment_std",
        "negative_review_share",
        "positive_review_share",
        "average_review_length",
        "review_days_since_latest",
        "review_span_days",
    )
    review_nlp = [
        column
        for column in columns
        if column.startswith(review_nlp_prefixes)
        or column.endswith("_keyword_rate")
        or (column.startswith("reviews_") and "_hash_" not in column)
    ]
    review_hash = [column for column in columns if column.startswith("reviews_hash_")]
    excluded = set(keys + targets + host_nlp + host_hash + review_nlp + review_hash)
    base = [column for column in columns if column not in excluded]
    return {
        "keys": keys,
        "targets": targets,
        "base_controls": base,
        "host_nlp": host_nlp,
        "host_hash": host_hash,
        "review_nlp": review_nlp,
        "review_hash": review_hash,
        "host_all": host_nlp + host_hash,
        "review_all": review_nlp + review_hash,
        "text_all": host_nlp + host_hash + review_nlp + review_hash,
        "model_all": base + host_nlp + host_hash + review_nlp + review_hash,
    }


def write_dataset(
    data: pd.DataFrame, output_dir: Path, name: str = "airbnb_features.parquet"
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    if name.endswith(".csv.gz") or name.endswith(".csv"):
        data.to_csv(path, index=False)
        return path
    data.to_parquet(path, index=False)
    return path


def write_feature_groups(
    data: pd.DataFrame, output_dir: Path, name: str = "airbnb_feature_groups.json"
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(feature_groups(data), indent=2) + "\n")
    return path
