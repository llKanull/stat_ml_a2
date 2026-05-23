# Airbnb Feature Construction

The processed modelling table has one row per listing at a snapshot date. Listing
and host metadata are taken from `listings.csv.gz`; review text is filtered to
reviews dated before the snapshot; demand targets are constructed from
`availability_90` in `listings.csv.gz`.

For temporal evaluation, build two separate tables:

- `airbnb_features_train.parquet`: all snapshots before each city's latest
  snapshot.
- `airbnb_features_test.parquet`: each city's latest snapshot.

This supports training and cross-validation on earlier data, then one final
evaluation on the newest available city snapshots.

Some archived June 2025 calendar files from Inside Airbnb are header-only files
with no rows, even after re-downloading. To keep train and test labels
consistent, the builder uses `availability_90` from `listings.csv.gz` for every
snapshot instead of mixing calendar-derived and listing-derived labels.

## Targets

- `target_unavailable_rate_90`: share of days in the next 90 days marked
  unavailable in the calendar.
- `target_high_demand`: top quartile of `target_unavailable_rate_90`.
- `target_reviews_ltm`, `target_reviews_per_month`, and
  `target_high_popularity` are retained as alternative popularity labels.

Calendar availability fields from `listings.csv.gz` are not included as model
features because they would duplicate the target horizon.

## Base Controls

Base features include capacity, price-derived ratios, host response metadata,
host verification, superhost status, instant booking, compact room/property type
signals, neighbourhood/property listing-count summaries, amenity group scores,
and distance from the city median coordinate.

The tabular construction goes beyond joining and one-hot encoding. We transform
the raw tables into a temporal listing-level panel:

- `listings.csv.gz` contributes property, host, amenity, location, review-count,
  and platform metadata at the cutoff snapshot.
- `reviews.csv.gz` is filtered to reviews strictly before the cutoff snapshot
  and aggregated into recency, volume, sentiment, topic, and language features.
- `availability_90` from `listings.csv.gz` is converted into 90-day
  unavailable-rate demand targets.

Constructed non-text features include:

- price per guest, beds per guest, bedrooms per guest, bathrooms per guest
- log price, log minimum nights, log host listing count, log prior reviews
- host tenure, days since first review, days since latest listing review
- host response strength and professional-host indicator
- entire-home, shared-room, and large-capacity flags
- semantic amenity scores for work, family, luxury, and safety amenities
- compact property-type flags for entire/private/shared, hotel-style, and
  unique stays
- neighbourhood and property-type listing counts within each city snapshot
- distance to the city median coordinate
- missingness indicators for price, response rates, rating, bedrooms,
  bathrooms, and beds
- forward 90-day unavailable and available rates as target-side summaries

These features are constructed from domain relationships between variables and
from aggregation across multiple raw files.

High-cardinality raw `property_type` and `neighbourhood_cleansed` one-hot
columns are intentionally omitted. They created hundreds of sparse columns in
`airbnb_features_latest.parquet` without adding much interpretable signal.

## Description NLP

`description` is cleaned for HTML, whitespace, and common mojibake. Constructed
features include presence, character/word/sentence counts, average sentence
length, VADER sentiment compound/positive/negative/neutral scores,
Flesch-Kincaid grade, and exclamation density.

## Host NLP

`host_about` receives the same compact text treatment as listing descriptions:
presence, length, sentence statistics, VADER sentiment, Flesch-Kincaid grade,
and exclamation density. Earlier hand-built lexicon scores and hashed
unigram/bigram features were removed to keep the modelling table smaller and
more interpretable.

## Review NLP

Reviews are filtered to dates before the snapshot and capped to the most recent
reviews per listing. Constructed features include review count before cutoff,
mean and standard deviation of VADER compound sentiment, mean positive/negative
sentiment proportions, mean Flesch-Kincaid grade, average review length, and
review recency/span.

The review features summarize guest tone, volume, recency, and language
complexity without adding hundreds of sparse keyword or hash columns.

## Leakage Control

All review text features are computed from reviews before the snapshot date.
Calendar availability from the forward horizon is used for the target, not as a
model input. `airbnb_feature_groups.json` separates keys, targets, base controls,
description NLP, host NLP, and review NLP.

## SBERT Embeddings

`notebooks/colab.ipynb` is the GPU path for Sentence-BERT semantic embeddings.
To avoid uploading the full raw data directory to Colab, first run
`scripts/build_sbert_inputs.py` locally. It writes compact text-only inputs:

- `data/processed/airbnb_sbert_input_train.parquet`
- `data/processed/airbnb_sbert_input_test.parquet`

The notebook reads those files and writes separate embedding outputs:

- `data/processed/airbnb_sbert_embeddings_train.parquet`
- `data/processed/airbnb_sbert_embeddings_test.parquet`

Each file is keyed by `city`, `snapshot`, and `listing_id` so experiments can
join embeddings only when they are needed.
