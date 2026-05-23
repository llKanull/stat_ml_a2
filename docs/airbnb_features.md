# Airbnb Feature Construction

The processed modelling table has one row per listing at a snapshot date. Listing
and host metadata are taken from `listings.csv.gz`; review text is filtered to
reviews dated before the snapshot; demand targets are constructed from
`calendar.csv.gz` over the forward 90-day horizon from the snapshot date.

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
host verification, superhost status, instant booking, room/property type,
neighbourhood, amenity group scores, and distance from the city median
coordinate.

The tabular construction goes beyond joining and one-hot encoding. We transform
the raw tables into a temporal listing-level panel:

- `listings.csv.gz` contributes property, host, amenity, location, review-count,
  and platform metadata at the cutoff snapshot.
- `reviews.csv.gz` is filtered to reviews strictly before the cutoff snapshot
  and aggregated into recency, volume, sentiment, topic, and language features.
- `calendar.csv.gz` is aggregated over the forward 90-day horizon to construct
  the demand target and target-side price/availability summaries.

Constructed non-text features include:

- price per guest, beds per guest, bedrooms per guest, bathrooms per guest
- log price, log minimum nights, log host listing count, log prior reviews
- host tenure, days since first review, days since latest listing review
- host response strength and professional-host indicator
- entire-home, shared-room, and large-capacity flags
- semantic amenity scores for work, family, luxury, and safety amenities
- distance to the city median coordinate
- missingness indicators for price, response rates, rating, bedrooms,
  bathrooms, and beds
- forward calendar unavailable rate, available rate, price mean/std, and
  weekend price premium as target-side summaries

These features are constructed from domain relationships between variables and
from aggregation across multiple raw files.

## Host NLP

`host_about` is cleaned for HTML, whitespace, and common mojibake. Constructed
features include profile presence/length, sentence statistics, lexical
diversity, sentiment balance, pronoun rates, punctuation counts, URL/year flags,
domain lexicon scores for warmth, professionalism, local expertise, travel,
arts/lifestyle, and family/home themes, plus hashed unigram/bigram features.

## Review NLP

Reviews are filtered to dates before the snapshot and capped to the most recent
reviews per listing. Constructed features include review count before cutoff,
mean sentiment, sentiment variability, positive/negative review shares, average
review length, recency/span, and keyword rates for cleanliness, location, host,
comfort, value, noise, check-in, amenities, and problems. Aggregated review text
also receives lexical summary features and hashed unigram/bigram features.

The review features summarize guest experience dimensions that are plausible
demand predictors: cleanliness, location convenience, host communication,
comfort, value, noise, check-in, and amenity problems.

## Leakage Control

All review text features are computed from reviews before the snapshot date.
Calendar availability from the forward horizon is used for the target, not as a
model input. `airbnb_feature_groups.json` separates keys, targets, base controls,
host NLP, review NLP, and hash features.

## SBERT Embeddings

`notebooks/colab.ipynb` is the GPU path for Sentence-BERT semantic embeddings.
It writes one normalized embedding for `host_about` and one for pre-cutoff review
text per listing. The output is keyed by `city`, `snapshot`, and `listing_id` so
it can be merged with the processed modelling table.
