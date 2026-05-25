from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from comp90051_project.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_f1,
    roc_auc_score,
)
from comp90051_project.models import CatBoostModel, FTTransformerModel, LogisticModel  # noqa: E402


# Best hyperparameters from CV
BEST_PARAMS: dict[str, dict[str, Any]] = {
    "Logistic": {"C": 0.001},
    "CatBoost": {"depth": 12, "iterations": 100},
    "FTTransformer": {"n_blocks": 6},
}

MODEL_FACTORIES = {
    "Logistic": lambda: LogisticModel(**BEST_PARAMS["Logistic"]),
    "CatBoost": lambda: CatBoostModel(**BEST_PARAMS["CatBoost"]),
    "FTTransformer": lambda: FTTransformerModel(**BEST_PARAMS["FTTransformer"]),
}


# ── Helpers ───────────────────────────────────────────────────────────────────


class _MedianImputer:
    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        self._meds = np.nanmedian(X, axis=0)
        return self._fill(X.copy())

    def transform(self, X):
        return self._fill(np.asarray(X, dtype=float).copy())

    def _fill(self, X):
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any():
                X[m, j] = self._meds[j]
        return X


def filter_active_short_term(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if "minimum_nights" in df.columns:
        mask &= df["minimum_nights"] <= 90
    if "has_availability" in df.columns:
        mask &= df["has_availability"].isin([True, "t", 1])
    if "number_of_reviews" in df.columns:
        mask &= df["number_of_reviews"] > 0
    return df[mask].reset_index(drop=True)


def select_numeric(df, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def merge_embeddings(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    embeddings = pd.read_parquet(path)
    return df.merge(
        embeddings,
        on=["city", "snapshot", "listing_id"],
        how="left",
        validate="one_to_one",
    )


def add_embedding_pca_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_gen: pd.DataFrame,
    *,
    raw_prefix: str,
    out_prefix: str,
    n_components: int = 16,
) -> list[str]:
    raw_cols = [c for c in df_train.columns if c.startswith(raw_prefix)]
    medians = df_train[raw_cols].median(axis=0)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(df_train[raw_cols].fillna(medians).to_numpy())

    out_cols = [f"{out_prefix}_pc_{i + 1:02d}" for i in range(n_components)]
    for frame in (df_train, df_test, df_gen):
        transformed = pca.transform(frame[raw_cols].fillna(medians).to_numpy())
        for i, col in enumerate(out_cols):
            frame[col] = transformed[:, i]

    print(f"  {out_prefix}: {len(raw_cols)} raw SBERT columns -> {len(out_cols)} PCA features")
    return out_cols


def evaluate(model, X_te, y_te, label):
    y_pred = model.predict(X_te)
    scores = model.predict_scores(X_te)
    acc = accuracy_score(y_te, y_pred)
    prf = precision_recall_f1(y_te, y_pred)
    auc = roc_auc_score(y_te, scores)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n    [{label}]")
    print(f"      Accuracy:   {acc:.4f}")
    print(f"      Macro-F1:   {float(prf['macro_f1']):.4f}")
    print(f"      Macro-Prec: {float(prf['macro_precision']):.4f}")
    print(f"      Macro-Rec:  {float(prf['macro_recall']):.4f}")
    print(f"      ROC-AUC:    {auc:.4f}")
    print("      Confusion matrix:")
    print(f"        TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
    print(f"        FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")

    return {
        "label": label,
        "n": len(y_te),
        "positive_rate": round(float(y_te.mean()), 4),
        "accuracy": round(acc, 6),
        "macro_f1": round(float(prf["macro_f1"]), 6),
        "macro_precision": round(float(prf["macro_precision"]), 6),
        "macro_recall": round(float(prf["macro_recall"]), 6),
        "auc": round(auc, 6),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-fttransformer", action="store_true")
    args = parser.parse_args()

    # Load and filter all splits
    print("Loading data …")
    df_train = filter_active_short_term(
        pd.read_parquet(args.data_dir / "airbnb_features_train.parquet")
    )
    df_test = filter_active_short_term(
        pd.read_parquet(args.data_dir / "airbnb_features_test.parquet")
    )
    df_gen = filter_active_short_term(
        pd.read_parquet(args.data_dir / "airbnb_features_generalization.parquet")
    )

    # Threshold computed on training set ONLY — applied to all splits
    RAW = "target_unavailable_rate_90"
    threshold = float(df_train[RAW].quantile(0.75))
    for df in (df_train, df_test, df_gen):
        df["target_high_demand"] = (df[RAW] >= threshold).astype(int)

    print(
        f"  Train: {len(df_train):,} rows  pos={df_train['target_high_demand'].mean():.3f}"
        f"  threshold={threshold:.4f}"
    )
    print(
        f"  Test:  {len(df_test):,} rows   pos={df_test['target_high_demand'].mean():.3f}"
        f"  cities={sorted(df_test['city'].unique())}"
    )
    print(
        f"  Gen:   {len(df_gen):,} rows   pos={df_gen['target_high_demand'].mean():.3f}"
        f"  cities={sorted(df_gen['city'].unique())}"
    )

    embedding_cols = []
    paths = {
        "train": args.data_dir / "airbnb_sbert_embeddings_train.parquet",
        "test": args.data_dir / "airbnb_sbert_embeddings_test.parquet",
        "generalization": args.data_dir / "airbnb_sbert_embeddings_generalization.parquet",
    }
    if all(path.exists() for path in paths.values()):
        print("\nLoading SBERT embeddings ...")
        df_train = merge_embeddings(df_train, paths["train"])
        df_test = merge_embeddings(df_test, paths["test"])
        df_gen = merge_embeddings(df_gen, paths["generalization"])
        host_embedding = add_embedding_pca_features(
            df_train,
            df_test,
            df_gen,
            raw_prefix="host_about_sbert_",
            out_prefix="host_sbert",
            n_components=16,
        )
        review_embedding = add_embedding_pca_features(
            df_train,
            df_test,
            df_gen,
            raw_prefix="reviews_sbert_",
            out_prefix="review_sbert",
            n_components=16,
        )
        embedding_cols = host_embedding + review_embedding
    else:
        print("\n  SBERT embedding parquets not found; skipping base_plus_text_embedding.")

    with open(args.data_dir / "airbnb_feature_groups.json") as f:
        groups = json.load(f)

    base = select_numeric(df_train, groups.get("base_controls", []))
    text = (
        select_numeric(df_train, groups.get("host_nlp", groups.get("host_all", [])))
        + select_numeric(df_train, groups.get("review_nlp", groups.get("review_all", [])))
        + select_numeric(df_train, groups.get("description_nlp", []))
    )

    feat_sets = {
        "base_only": base,
        "base_plus_text": base + text,
    }
    if embedding_cols:
        feat_sets["base_plus_text_embedding"] = base + text + embedding_cols

    y_train = df_train["target_high_demand"].to_numpy()
    y_test = df_test["target_high_demand"].to_numpy()
    y_gen = df_gen["target_high_demand"].to_numpy()

    # Models to run
    model_names = ["Logistic", "CatBoost"]
    if not args.skip_fttransformer:
        model_names.append("FTTransformer")
    else:
        print("\n  (FT-Transformer skipped via --skip-fttransformer)")

    # Run evaluation
    all_records = []

    print(f"\n{'═' * 65}")
    print("  FINAL EVALUATION — TEMPORAL TEST + GEOGRAPHIC GENERALISATION")
    print(f"{'═' * 65}")
    print("  Threshold from training set only. Test/Gen sets never touched before.")

    for feat_name, feat_cols in feat_sets.items():
        X_tr = df_train[feat_cols].to_numpy()
        X_te = df_test[feat_cols].to_numpy()
        X_ge = df_gen[feat_cols].to_numpy()

        for model_name in model_names:
            print(f"\n  {'─' * 60}")
            print(f"  {model_name}  |  {feat_name}  ({len(feat_cols)} features)")
            print(f"  {'─' * 60}")

            # Impute
            imp = _MedianImputer()
            X_tr_imp = imp.fit_transform(X_tr)
            X_te_imp = imp.transform(X_te)
            X_ge_imp = imp.transform(X_ge)

            # Fit on full training set with best CV hyperparameters
            model = MODEL_FACTORIES[model_name]()
            model.fit(X_tr_imp, y_train)

            # Temporal test
            rec_test = evaluate(
                model,
                X_te_imp,
                y_test,
                f"Temporal test  n={len(y_test):,}  (training cities, Jan 2026)",
            )
            rec_test.update({"model": model_name, "features": feat_name, "split": "temporal_test"})
            all_records.append(rec_test)

            # Geographic generalisation
            rec_gen = evaluate(
                model, X_ge_imp, y_gen, f"Geographic gen n={len(y_gen):,}  (Sydney / Tasmania / WA)"
            )
            rec_gen.update({"model": model_name, "features": feat_name, "split": "geographic_gen"})
            all_records.append(rec_gen)

    # Save
    out_dir = args.output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_set_results.csv"
    pd.DataFrame(all_records).to_csv(out_path, index=False)

    # Print summary
    print(f"\n{'═' * 65}")
    print("  SUMMARY")
    print(f"{'═' * 65}")
    header = f"  {'Model':14s} {'Features':15s} {'Split':16s}  {'AUC':>6s}  {'F1':>6s}  {'Acc':>6s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in all_records:
        print(
            f"  {r['model']:14s} {r['features']:15s} {r['split']:16s}"
            f"  {r['auc']:.4f}  {r['macro_f1']:.4f}  {r['accuracy']:.4f}"
        )

    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
