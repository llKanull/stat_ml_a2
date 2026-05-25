#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full COMP90051 project pipeline
#
# Usage:
#   bash scripts/run_pipeline.sh                           # full run
#   bash scripts/run_pipeline.sh --skip-download           # skip download
#   bash scripts/run_pipeline.sh --skip-build              # skip download+build
#   bash scripts/run_pipeline.sh --skip-fttransformer      # skip FT-Transformer
#   bash scripts/run_pipeline.sh --fast                    # 5-fold, no FT-Trans
#   bash scripts/run_pipeline.sh --skip-download --skip-fttransformer
#
# Run from project root: cd /path/to/stat_ml_a2
# =============================================================================

set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
SKIP_BUILD=false
SKIP_FTT=false
FAST=false

for arg in "$@"; do
  case $arg in
    --skip-download)      SKIP_DOWNLOAD=true ;;
    --skip-build)         SKIP_BUILD=true; SKIP_DOWNLOAD=true ;;
    --skip-fttransformer) SKIP_FTT=true ;;
    --fast)               FAST=true; SKIP_FTT=true ;;
    *) echo "[ERROR] Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD="\033[1m"; GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; RESET="\033[0m"
step() { echo -e "\n${BOLD}${GREEN}══ $1 ${RESET}"; }
info() { echo -e "${YELLOW}   $1${RESET}"; }
ok()   { echo -e "${GREEN}   ✓ $1${RESET}"; }
skip() { echo -e "${YELLOW}   ↷ Skipping: $1${RESET}"; }
START_TIME=$SECONDS

# ── Sanity check ──────────────────────────────────────────────────────────────
if [ ! -f "scripts/run_experiment.py" ]; then
  echo -e "${RED}[ERROR] Run from project root.${RESET}"
  echo "  cd /path/to/stat_ml_a2 && bash scripts/run_pipeline.sh"
  exit 1
fi

echo -e "\n${BOLD}COMP90051 Project Pipeline${RESET}"
echo "  Started:            $(date)"
echo "  Skip download:      $SKIP_DOWNLOAD"
echo "  Skip build:         $SKIP_BUILD"
echo "  Skip FT-Transformer:$SKIP_FTT"
echo "  Fast mode:          $FAST"

# ── Detect existing files ─────────────────────────────────────────────────────
HAS_RAW_DATA=false
HAS_BUILD_PARQUETS=false
HAS_SPLIT_PARQUETS=false

[ -d "data/raw/inside_airbnb_australia" ] && HAS_RAW_DATA=true
[ -f "data/processed/airbnb_features_previous_all.parquet" ] && \
  [ -f "data/processed/airbnb_features_latest.parquet" ] && HAS_BUILD_PARQUETS=true
[ -f "data/processed/airbnb_features_train.parquet" ] && \
  [ -f "data/processed/airbnb_feature_groups.json" ] && HAS_SPLIT_PARQUETS=true


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

if [ "$SKIP_BUILD" = true ] && [ "$HAS_SPLIT_PARQUETS" = true ]; then
  skip "feature build — using existing split parquets"
  HAS_BUILD_PARQUETS=true
elif [ "$SKIP_BUILD" = true ] && [ "$HAS_BUILD_PARQUETS" = true ]; then
  skip "feature build — build parquets already exist"
elif [ "$SKIP_BUILD" = true ]; then
  echo -e "${RED}[ERROR] --skip-build set but no usable parquets found.${RESET}"
  exit 1
else
  info "Building features (includes NLP — may take 15–30 min) …"

  python scripts/build_airbnb_features.py \
    --snapshot-split before-latest \
    --output-name airbnb_features_previous_all.parquet

  ok "Previous snapshots → data/processed/airbnb_features_previous_all.parquet"

  python scripts/build_airbnb_features.py \
    --snapshot-split latest \
    --output-name airbnb_features_latest.parquet

  ok "Latest snapshots   → data/processed/airbnb_features_latest.parquet"
  HAS_BUILD_PARQUETS=true
fi


# ── Step 3: Create splits ─────────────────────────────────────────────────────
step "Step 3/4: Create train / test / generalisation splits"

if [ "$HAS_SPLIT_PARQUETS" = true ] && [ "$SKIP_BUILD" = true ]; then
  skip "split creation — existing splits retained (re-run without --skip-build to rebuild)"
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

OUTER_K=10
if [ "$FAST" = true ]; then
  OUTER_K=5
  info "Fast mode: 5-fold CV"
else
  info "Full run: 10-fold CV"
fi

if [ "$SKIP_FTT" = true ]; then
  info "FT-Transformer skipped (run on Colab GPU separately)"
  python scripts/run_experiment.py \
    --outer-k $OUTER_K \
    --inner-k 3 \
    --skip-fttransformer
else
  info "Running all 3 models including FT-Transformer (~2–4 hours on GPU)"
  python scripts/run_experiment.py \
    --outer-k $OUTER_K \
    --inner-k 3
fi

ok "Experiment complete"


# ── Step 5: Test set evaluation ───────────────────────────────────────────────
step "Step 5/5: Test set & geographic generalisation evaluation"

if [ "$SKIP_FTT" = true ]; then
  python scripts/evaluate_test_set.py --skip-fttransformer
else
  python scripts/evaluate_test_set.py
fi

ok "Test evaluation complete"
info "  outputs/tables/test_set_results.csv"
info "  outputs/tables/fold_scores.csv"
info "  outputs/tables/results_summary.csv"


# ── Done ──────────────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - START_TIME ))
echo -e "\n${BOLD}${GREEN}Pipeline complete!${RESET}"
echo "  Finished: $(date)"
printf "  Total time: %dh %02dm %02ds\n\n" \
  $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 ))