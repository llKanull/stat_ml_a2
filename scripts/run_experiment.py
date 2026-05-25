from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from comp90051_project.cv import k_fold_indices  # noqa: E402
from comp90051_project.metrics import (  # noqa: E402
    accuracy_score,
    mean_and_error_bar,
    precision_recall_f1,
    roc_auc_score,
)
from comp90051_project.models import CatBoostModel, FTTransformerModel, LogisticModel  # noqa: E402
from comp90051_project.tuning import tune_hyperparameters  # noqa: E402


# Hyperparameter grids
LOGISTIC_GRID: dict = {"C": [0.0001, 0.001, 0.01]}
CATBOOST_GRID: dict = {"depth": [9, 12, 15]}
FTTRANSFORMER_GRID: dict = {"n_blocks": [4, 6, 8]}


class _MedianImputer:
    def __init__(self) -> None:
        self._medians: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self._medians = np.nanmedian(X, axis=0)
        return self._fill(X.copy())

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._medians is None:
            raise RuntimeError("Call fit_transform before transform.")
        return self._fill(np.asarray(X, dtype=float).copy())

    def _fill(self, X: np.ndarray) -> np.ndarray:
        if self._medians is None:
            raise RuntimeError("Call fit_transform before transform.")
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = self._medians[j]
        return X


# Single experiment
def _run_combo(
    *,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    factory,
    param_grid: dict,
    feat_name: str,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    n_inner: int,
) -> list[dict]:
    """Run one (model × feature-set) combination across all outer folds."""
    fold_records: list[dict] = []

    for fold_i, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        t0 = time.perf_counter()

        X_tr_raw = X[outer_train_idx]
        X_te_raw = X[outer_test_idx]
        y_tr = y[outer_train_idx]
        y_te = y[outer_test_idx]

        # Impute NaN — fit on outer train only to prevent leakage
        imputer = _MedianImputer()
        X_tr = imputer.fit_transform(X_tr_raw)
        X_te = imputer.transform(X_te_raw)

        # Inner CV for hyperparameter selection (macro-F1 criterion)
        inner_splits = list(k_fold_indices(len(X_tr), k=n_inner, random_state=fold_i * 17))

        def _score_fn(estimator, Xtr, ytr, Xval, yval):
            y_pred = estimator.predict(Xval)
            return float(precision_recall_f1(yval, y_pred)["macro_f1"])

        best_params, _ = tune_hyperparameters(
            X_tr,
            y_tr,
            estimator_factory=factory,
            param_grid=param_grid,
            inner_splits=inner_splits,
            score_function=_score_fn,
        )

        # Train with best params on full outer training fold
        model = factory(best_params)
        model.fit(X_tr, y_tr)

        # Evaluate on held-out outer test fold
        y_pred = model.predict(X_te)
        scores = model.predict_scores(X_te)

        acc = accuracy_score(y_te, y_pred)
        prf = precision_recall_f1(y_te, y_pred)
        auc = roc_auc_score(y_te, scores)
        elapsed = time.perf_counter() - t0

        record = {
            "model": model_name,
            "features": feat_name,
            "fold": fold_i + 1,
            "best_params": json.dumps(best_params),
            "accuracy": round(acc, 6),
            "macro_f1": round(float(prf["macro_f1"]), 6),
            "macro_precision": round(float(prf["macro_precision"]), 6),
            "macro_recall": round(float(prf["macro_recall"]), 6),
            "auc": round(auc, 6),
            "elapsed_s": round(elapsed, 1),
        }
        fold_records.append(record)

        print(
            f"  [{model_name:12s} | {feat_name:15s} | fold {fold_i + 1:2d}]"
            f"  acc={acc:.3f}  f1={float(prf['macro_f1']):.3f}"
            f"  auc={auc:.3f}  {elapsed:.1f}s  best={best_params}"
        )

    return fold_records


# Summary
def _summarise(fold_records: list[dict]) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "auc"]
    rows = []
    df = pd.DataFrame(fold_records)
    for _, group in df.groupby(["model", "features"]):
        row: dict = {
            "model": group["model"].iloc[0],
            "features": group["features"].iloc[0],
        }
        for metric in metrics:
            mean, ci = mean_and_error_bar(group[metric].to_numpy())
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_ci95"] = round(ci, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _select_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _add_embedding_pca_features(
    df: pd.DataFrame,
    data_dir: Path,
    *,
    n_components: int = 16,
) -> list[str]:
    train_embedding_path = data_dir / "airbnb_sbert_embeddings_train.parquet"
    if not train_embedding_path.exists():
        print("  SBERT train embedding parquet not found; skipping embedding feature set.")
        return []

    embeddings = pd.read_parquet(train_embedding_path)
    df_with_embeddings = df.merge(
        embeddings,
        on=["city", "snapshot", "listing_id"],
        how="left",
        validate="one_to_one",
    )

    embedding_cols: list[str] = []
    for raw_prefix, out_prefix in [
        ("host_about_sbert_", "host_sbert"),
        ("reviews_sbert_", "review_sbert"),
    ]:
        raw_cols = [c for c in df_with_embeddings.columns if c.startswith(raw_prefix)]
        medians = df_with_embeddings[raw_cols].median(axis=0)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        transformed = pca.fit_transform(df_with_embeddings[raw_cols].fillna(medians).to_numpy())
        out_cols = [f"{out_prefix}_pc_{i + 1:02d}" for i in range(n_components)]
        for i, col in enumerate(out_cols):
            df[col] = transformed[:, i]
        print(f"  {out_prefix}: {len(raw_cols)} raw SBERT columns -> {len(out_cols)} PCA features")
        embedding_cols.extend(out_cols)

    return embedding_cols


# Main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--outer-k", type=int, default=10, help="Outer CV folds (spec requires >= 10)."
    )
    parser.add_argument(
        "--inner-k", type=int, default=3, help="Inner CV folds for tuning (spec requires >= 3)."
    )
    parser.add_argument("--target", default="target_high_demand")
    parser.add_argument(
        "--skip-fttransformer",
        action="store_true",
        help="Skip FT-Transformer (fast dev: Logistic + CatBoost only).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load data
    train_path = args.data_dir / "airbnb_features_train.parquet"
    groups_path = args.data_dir / "airbnb_feature_groups.json"

    for p in (train_path, groups_path):
        if not p.exists():
            sys.exit(
                f"\n[ERROR] Not found: {p}\n"
                "Run the pipeline first:\n"
                "  python scripts/import_airbnb.py\n"
                "  python scripts/build_airbnb_features.py\n"
                "  python scripts/create_airbnb_eval_splits.py\n"
            )

    print(f"\nLoading {train_path} …")
    df = pd.read_parquet(train_path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    with groups_path.open() as fh:
        groups = json.load(fh)

    # Target
    if args.target not in df.columns:
        sys.exit(
            f"\n[ERROR] Target '{args.target}' not in columns.\n"
            f"Available: {[c for c in df.columns if c.startswith('target_')]}\n"
        )
    y = df[args.target].to_numpy()
    print(f"  Target: {args.target}  positive_rate={y.mean():.3f}")

    # Feature sets
    base = _select_numeric(df, groups.get("base_controls", []))
    host = _select_numeric(df, groups.get("host_nlp", groups.get("host_all", [])))
    rev = _select_numeric(df, groups.get("review_nlp", groups.get("review_all", [])))
    desc = _select_numeric(df, groups.get("description_nlp", []))
    text = host + rev + desc
    embedding = _add_embedding_pca_features(df, args.data_dir, n_components=16)

    feature_sets: dict[str, list[str]] = {
        "base_only": base,
        "base_plus_text": base + text,
    }
    if embedding:
        feature_sets["base_plus_text_embedding"] = base + text + embedding
    for name, cols in feature_sets.items():
        print(f"  Feature set '{name}': {len(cols)} columns")

    if not base:
        sys.exit("\n[ERROR] base_controls is empty — check feature_groups.json\n")

    # Outer CV splits
    outer_splits = list(k_fold_indices(len(df), k=args.outer_k, random_state=args.random_state))
    print(f"\nOuter CV: {args.outer_k}-fold  |  Inner CV: {args.inner_k}-fold\n")

    # Model registry
    all_models: list[tuple[str, object, dict]] = [
        (
            "Logistic",
            lambda p: LogisticModel(**p),
            LOGISTIC_GRID,
        ),
        (
            "CatBoost",
            lambda p: CatBoostModel(iterations=200, **p),
            CATBOOST_GRID,
        ),
    ]
    if not args.skip_fttransformer:
        all_models.append(
            (
                "FTTransformer",
                lambda p: FTTransformerModel(**p),
                FTTRANSFORMER_GRID,
            )
        )
    else:
        print("  (FT-Transformer skipped via --skip-fttransformer)\n")

    # Run
    all_fold_records: list[dict] = []

    for feat_name, feat_cols in feature_sets.items():
        X = df[feat_cols].to_numpy()
        for model_name, factory, param_grid in all_models:
            print(f"\n{'─' * 72}")
            print(f"  Model: {model_name}   Features: {feat_name}")
            print(f"{'─' * 72}")
            records = _run_combo(
                X=X,
                y=y,
                model_name=model_name,
                factory=factory,
                param_grid=param_grid,
                feat_name=feat_name,
                outer_splits=outer_splits,
                n_inner=args.inner_k,
            )
            all_fold_records.extend(records)

    # Save
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(all_fold_records)
    fold_path = tables_dir / "fold_scores.csv"
    fold_df.to_csv(fold_path, index=False)

    summary_df = _summarise(all_fold_records)
    summary_path = tables_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    print(f"\n{'═' * 72}")
    print("  RESULTS SUMMARY  (mean ± 95 % CI, outer folds)")
    print(f"{'═' * 72}")
    header = (
        f"{'Model':14s}  {'Features':15s}  {'Accuracy':>15s}  {'Macro-F1':>15s}  {'ROC-AUC':>15s}"
    )
    print(header)
    print("─" * len(header))

    for _, row in summary_df.iterrows():

        def fmt(m):
            return f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_ci95']:.3f}"

        print(
            f"{row['model']:14s}  {row['features']:15s}"
            f"  {fmt('accuracy'):>15s}  {fmt('macro_f1'):>15s}  {fmt('auc'):>15s}"
        )

    print(f"\nFold scores  → {fold_path}")
    print(f"Summary      → {summary_path}")


if __name__ == "__main__":
    main()
