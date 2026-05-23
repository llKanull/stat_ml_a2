from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    SentimentIntensityAnalyzer = None

try:
    import textstat
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    textstat = None

_VADER_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None


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
    snapshot_split: str = "all"


def load_manifest(raw_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(raw_dir / "manifest.csv")
    return manifest[manifest["file_name"].isin({"listings.csv.gz", "reviews.csv.gz"})].copy()


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


def syllable_count(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text: str, tokens: list[str], sentence_count: int) -> float:
    if len(tokens) < 5:
        return np.nan
    if textstat is not None:
        return float(textstat.flesch_kincaid_grade(text))
    syllables = sum(syllable_count(token) for token in tokens)
    return 0.39 * (len(tokens) / sentence_count) + 11.8 * (syllables / len(tokens)) - 15.59


def sentiment_scores(text: str, tokens: list[str]) -> dict[str, float]:
    if _VADER_ANALYZER is not None:
        scores = _VADER_ANALYZER.polarity_scores(text)
        return {
            "compound": float(scores["compound"]),
            "positive": float(scores["pos"]),
            "negative": float(scores["neg"]),
            "neutral": float(scores["neu"]),
        }
    positive_count = sum(token in POSITIVE_WORDS for token in tokens)
    negative_count = sum(token in NEGATIVE_WORDS for token in tokens)
    token_count = len(tokens)
    balance = (positive_count - negative_count) / token_count if token_count else 0.0
    return {
        "compound": float(np.clip(balance, -1.0, 1.0)),
        "positive": positive_count / token_count if token_count else 0.0,
        "negative": negative_count / token_count if token_count else 0.0,
        "neutral": 1.0 - ((positive_count + negative_count) / token_count) if token_count else 0.0,
    }


def text_features(texts: pd.Series, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for text in texts.fillna("").map(clean_text):
        tokens = tokenise(text)
        token_count = len(tokens)
        sentence_count = max(1, len([part for part in SENTENCE_RE.split(text) if part.strip()]))
        sentiment = sentiment_scores(text, tokens)

        row = {
            f"{prefix}_present": float(bool(text)),
            f"{prefix}_char_count": float(len(text)),
            f"{prefix}_word_count": float(token_count),
            f"{prefix}_sentence_count": float(sentence_count),
            f"{prefix}_avg_sentence_words": token_count / sentence_count if token_count else 0.0,
            f"{prefix}_sentiment_compound": sentiment["compound"],
            f"{prefix}_sentiment_positive": sentiment["positive"],
            f"{prefix}_sentiment_negative": sentiment["negative"],
            f"{prefix}_sentiment_neutral": sentiment["neutral"],
            f"{prefix}_readability_fk_grade": flesch_kincaid_grade(text, tokens, sentence_count),
            f"{prefix}_exclamation_density": text.count("!") / token_count if token_count else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows, index=texts.index)


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


def categorical_summary_features(listings: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=listings.index)

    if "room_type" in listings:
        encoded = pd.get_dummies(
            listings["room_type"].fillna("missing"), prefix="room_type", dtype=float
        )
        out = pd.concat([out, encoded], axis=1)

    if "host_response_time" in listings:
        encoded = pd.get_dummies(
            listings["host_response_time"].fillna("missing"),
            prefix="host_response_time",
            dtype=float,
        )
        out = pd.concat([out, encoded], axis=1)

    if "property_type" in listings:
        property_type = listings["property_type"].fillna("").str.lower()
        out["property_entire_flag"] = property_type.str.contains(r"\bentire\b", regex=True).astype(
            float
        )
        out["property_private_room_flag"] = property_type.str.contains("private room").astype(float)
        out["property_shared_room_flag"] = property_type.str.contains("shared room").astype(float)
        out["property_hotel_flag"] = property_type.str.contains(
            "hotel|hostel|resort|aparthotel"
        ).astype(float)
        out["property_unique_stay_flag"] = property_type.str.contains(
            "boat|bus|camper|castle|cave|container|farm|hut|tiny|train|treehouse|yurt"
        ).astype(float)

    for column in ["neighbourhood_cleansed", "property_type"]:
        if column in listings:
            counts = listings.groupby(["city", "snapshot", column])["id"].transform("count")
            out[f"{column}_listing_count"] = counts.astype(float)
            out[f"log_{column}_listing_count"] = np.log1p(counts)

    return out


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

    out = pd.concat([out, categorical_summary_features(listings)], axis=1)

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
    return text_features(texts, "host_about")


def description_features(listings: pd.DataFrame) -> pd.DataFrame:
    texts = listings.get("description", pd.Series("", index=listings.index))
    return text_features(texts, "description")


def review_text_scores(text: str) -> dict[str, float]:
    tokens = tokenise(text)
    token_count = len(tokens)
    sentence_count = max(1, len([part for part in SENTENCE_RE.split(text) if part.strip()]))
    sentiment = sentiment_scores(text, tokens)
    scores = {
        "review_word_count": float(token_count),
        "review_sentiment_compound": sentiment["compound"],
        "review_sentiment_positive": sentiment["positive"],
        "review_sentiment_negative": sentiment["negative"],
        "review_readability_fk_grade": flesch_kincaid_grade(text, tokens, sentence_count),
    }
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

    agg_spec = {
        "review_text_count": ("clean_comments", "size"),
        "review_sentiment_compound_mean": ("review_sentiment_compound", "mean"),
        "review_sentiment_compound_std": ("review_sentiment_compound", "std"),
        "review_sentiment_positive_mean": ("review_sentiment_positive", "mean"),
        "review_sentiment_negative_mean": ("review_sentiment_negative", "mean"),
        "review_readability_fk_grade_mean": ("review_readability_fk_grade", "mean"),
        "average_review_length": ("review_word_count", "mean"),
    }
    counts = reviews.groupby("listing_id").agg(**agg_spec)
    counts = counts.rename(columns={"review_text_count": "review_count_before_cutoff"})
    latest = reviews.groupby("listing_id")["date"].max()
    earliest = reviews.groupby("listing_id")["date"].min()
    counts["review_days_since_latest"] = (pd.Timestamp(snapshot) - latest).dt.days
    counts["review_span_days"] = (latest - earliest).dt.days
    counts["review_sentiment_compound_std"] = counts["review_sentiment_compound_std"].fillna(0.0)

    return counts.reset_index()


def read_listings(path: Path, city: str, snapshot: str) -> pd.DataFrame:
    listings = pd.read_csv(path, low_memory=False)
    listings["city"] = city
    listings["snapshot"] = snapshot
    return listings


def read_reviews(path: Path, max_rows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def listing_availability_targets(listings: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    if horizon_days != 90 or "availability_90" not in listings:
        return pd.DataFrame(columns=["listing_id"])
    available_days = numeric_series(listings, "availability_90").clip(lower=0, upper=90)
    targets = pd.DataFrame(
        {
            "listing_id": listings["id"].astype("int64"),
            "target_horizon_days": float(horizon_days),
            "target_available_rate_90": available_days / horizon_days,
        },
        index=listings.index,
    )
    targets["target_unavailable_rate_90"] = 1.0 - targets["target_available_rate_90"]
    return targets


def build_snapshot_features(
    listings_path: Path,
    reviews_path: Path,
    city: str,
    snapshot: str,
    config: BuildConfig,
) -> pd.DataFrame:
    listings = read_listings(listings_path, city, snapshot)
    listing_df = listing_features(listings)
    description_df = description_features(listings)
    host_df = host_about_features(listings)
    reviews = read_reviews(reviews_path, config.max_review_rows)
    review_df = review_aggregates(reviews, config.max_reviews_per_listing, snapshot)
    target_df = listing_availability_targets(listings, config.target_horizon_days)

    features = pd.concat([listing_df, description_df, host_df], axis=1)
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
    if config.snapshot_split == "latest":
        grouped = grouped.sort_values(["city", "snapshot"]).groupby("city").tail(1)
    elif config.snapshot_split == "before-latest":
        latest = grouped.groupby("city")["snapshot"].transform("max")
        grouped = grouped[grouped["snapshot"] < latest]
    elif config.snapshot_split == "previous":
        latest = grouped.groupby("city")["snapshot"].transform("max")
        grouped = grouped[grouped["snapshot"] < latest].sort_values(["city", "snapshot"])
        grouped = grouped.groupby("city").tail(1)
    elif config.snapshot_split != "all":
        raise ValueError("snapshot_split must be one of: all, latest, before-latest, previous")
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
        if listings_path.exists() and reviews_path.exists():
            records.append(
                build_snapshot_features(listings_path, reviews_path, city, snapshot, config)
            )

    if not records:
        raise FileNotFoundError(f"No listing/review snapshots found under {config.raw_dir}")

    data = pd.concat(records, ignore_index=True, sort=False)
    high_demand = (
        data["target_unavailable_rate_90"] >= data["target_unavailable_rate_90"].quantile(0.75)
    ).astype(int)
    high_popularity = (
        data["target_reviews_ltm"]
        >= max(config.min_reviews_ltm_for_popular, data["target_reviews_ltm"].quantile(0.75))
    ).astype(int)
    data = data.assign(
        target_high_demand=high_demand,
        target_high_popularity=high_popularity,
    )
    return data.replace([np.inf, -np.inf], np.nan)


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(data.columns)
    keys = ["city", "snapshot", "listing_id"]
    targets = [column for column in columns if column.startswith("target_")]
    description_nlp = [column for column in columns if column.startswith("description_")]
    host_nlp = [column for column in columns if column.startswith("host_about_")]
    review_nlp = [
        column
        for column in columns
        if column.startswith("review_") or column.startswith("average_review_length")
    ]
    excluded = set(keys + targets + description_nlp + host_nlp + review_nlp)
    base = [column for column in columns if column not in excluded]
    return {
        "keys": keys,
        "targets": targets,
        "base_controls": base,
        "description_nlp": description_nlp,
        "host_nlp": host_nlp,
        "review_nlp": review_nlp,
        "host_all": host_nlp,
        "review_all": review_nlp,
        "text_all": description_nlp + host_nlp + review_nlp,
        "model_all": base + description_nlp + host_nlp + review_nlp,
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
