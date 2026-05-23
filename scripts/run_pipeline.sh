#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full COMP90051 project pipeline
#
# Usage:
#   bash scripts/run_pipeline.sh                  # full run (all steps)
#   bash scripts/run_pipeline.sh --skip-download  # skip download only
#   bash scripts/run_pipeline.sh --skip-build     # skip download + build
#   bash scripts/run_pipeline.sh --fast           # 5-fold, no FT-Transformer
#   bash scripts/run_pipeline.sh --fast --skip-build
#
# Run from project root:  cd /path/to/stat_ml_a2
# =============================================================================

set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
SKIP_BUILD=false
FAST=false

for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=true ;;
    --skip-build)    SKIP_BUILD=true; SKIP_DOWNLOAD=true ;;
    --fast)          FAST=true ;;
    *) echo "[ERROR] Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD="\033[1m"; GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; RESET="\033[0m"
step()  { echo -e "\n${BOLD}${GREEN}══ $1 ${RESET}"; }
info()  { echo -e "${YELLOW}   $1${RESET}"; }
ok()    { echo -e "${GREEN}   ✓ $1${RESET}"; }
skip()  { echo -e "${YELLOW}   ↷ Skipping: $1${RESET}"; }
START_TIME=$SECONDS

# ── Sanity check ──────────────────────────────────────────────────────────────
if [ ! -f "scripts/run_experiment.py" ]; then
  echo -e "${RED}[ERROR] Run from project root.${RESET}"
  echo "  cd /path/to/stat_ml_a2 && bash scripts/run_pipeline.sh"
  exit 1
fi

echo -e "\n${BOLD}COMP90051 Project Pipeline${RESET}"
echo "  Started:      $(date)"
echo "  Fast mode:    $FAST"
echo "  Skip build:   $SKIP_BUILD"

# ── Check what files already exist ────────────────────────────────────────────
HAS_RAW_DATA=false
HAS_BUILD_PARQUETS=false
HAS_SPLIT_PARQUETS=false

[ -d "data/raw/inside_airbnb_australia" ] && HAS_RAW_DATA=true
[ -f "data/processed/airbnb_features_previous_all.parquet" ] && \
  [ -f "data/processed/airbnb_features_latest.parquet" ] && HAS_BUILD_PARQUETS=true
[ -f "data/processed/airbnb_features_train.parquet" ] && \
  [ -f "data/processed/airbnb_feature_groups.json" ] && HAS_SPLIT_PARQUETS=true

echo "  Has raw data: $HAS_RAW_DATA"
echo "  Has build parquets: $HAS_BUILD_PARQUETS"
echo "  Has split parquets: $HAS_SPLIT_PARQUETS"


# ── Step 1: Download ──────────────────────────────────────────────────────────
step "Step 1/4: Download Inside Airbnb data"

if [ "$SKIP_DOWNLOAD" = true ] || [ "$HAS_RAW_DATA" = true ]; then
  skip "download"
else
  info "Downloading Australian Inside Airbnb snapshots (10–20 min) …"
  python scripts/import_airbnb.py
  ok "Download complete"
fi


# ── Step 2: Build features ────────────────────────────────────────────────────
step "Step 2/4: Build feature parquets"

if [ "$SKIP_BUILD" = true ] || [ "$HAS_SPLIT_PARQUETS" = true ]; then
  # If we already have the final split parquets, we can skip both build AND split
  if [ "$HAS_SPLIT_PARQUETS" = true ] && [ "$HAS_BUILD_PARQUETS" = false ]; then
    skip "feature build — final split parquets already exist"
    HAS_BUILD_PARQUETS=true  # signal step 3 to skip too
  elif [ "$HAS_BUILD_PARQUETS" = true ]; then
    skip "feature build — build parquets already exist"
  else
    echo -e "${RED}[ERROR] --skip-build set but no usable parquets found.${RESET}"
    echo "  Run without --skip-build to build from raw data."
    exit 1
  fi
else
  info "Building features for all-but-latest snapshots (training data, ~15–30 min) …"
  python scripts/build_airbnb_features.py \
    --snapshot-split before-latest \
    --output-name airbnb_features_previous_all.parquet
  ok "Previous snapshots built → data/processed/airbnb_features_previous_all.parquet"

  info "Building features for latest snapshot per city (test data) …"
  python scripts/build_airbnb_features.py \
    --snapshot-split latest \
    --output-name airbnb_features_latest.parquet
  ok "Latest snapshots built → data/processed/airbnb_features_latest.parquet"
  HAS_BUILD_PARQUETS=true
fi


# ── Step 3: Create splits ─────────────────────────────────────────────────────
step "Step 3/4: Create train / test / generalisation splits"

# Skip if: final split parquets already exist AND we didn't just rebuild
if [ "$HAS_SPLIT_PARQUETS" = true ] && [ "$HAS_BUILD_PARQUETS" = true ] && [ "$SKIP_BUILD" = true ]; then
  skip "split creation — airbnb_features_train.parquet already exists"
else
  python scripts/create_airbnb_eval_splits.py
  ok "Splits written:"
  info "  data/processed/airbnb_features_train.parquet"
  info "  data/processed/airbnb_features_test.parquet"
  info "  data/processed/airbnb_features_generalization.parquet"
  info "  data/processed/airbnb_feature_groups.json"
fi


# ── Step 4: Run experiment ────────────────────────────────────────────────────
step "Step 4/4: Run experiment"

if [ "$FAST" = true ]; then
  info "Fast mode: 5-fold, Logistic + CatBoost only (no FT-Transformer)"
  python scripts/run_experiment.py \
    --outer-k 5 \
    --inner-k 3 \
    --skip-fttransformer
else
  info "Full run: 10-fold, all three models (~2–4 hours on CPU)"
  python scripts/run_experiment.py \
    --outer-k 10 \
    --inner-k 3
fi

ok "Experiment complete"
info "  outputs/tables/fold_scores.csv"
info "  outputs/tables/results_summary.csv"


# ── Done ──────────────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - START_TIME ))
echo -e "\n${BOLD}${GREEN}Pipeline complete!${RESET}"
echo "  Finished: $(date)"
printf "  Total time: %dh %02dm %02ds\n\n" \
  $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 ))
