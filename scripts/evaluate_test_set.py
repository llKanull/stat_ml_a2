from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
 
from comp90051_project.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_f1,
    roc_auc_score,
)
from comp90051_project.models import CatBoostModel, FTTransformerModel, LogisticModel
 
 
# Best hyperparameters from CV
BEST_PARAMS = {
    "Logistic":      {"C": 0.001},
    "CatBoost":      {"depth": 12, "iterations": 100},
    "FTTransformer": {"n_blocks": 6},
}
 
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
            if m.any(): X[m, j] = self._meds[j]
        return X
 
 
def filter_active_short_term(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if "minimum_nights"    in df.columns: mask &= df["minimum_nights"] <= 90
    if "has_availability"  in df.columns: mask &= df["has_availability"].isin([True,"t",1])
    if "number_of_reviews" in df.columns: mask &= df["number_of_reviews"] > 0
    return df[mask].reset_index(drop=True)
 
 
def select_numeric(df, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
 
 
def evaluate(model, X_te, y_te, label):
    y_pred = model.predict(X_te)
    scores = model.predict_scores(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prf  = precision_recall_f1(y_te, y_pred)
    auc  = roc_auc_score(y_te, scores)
    cm   = confusion_matrix(y_te, y_pred)
 
    print(f"\n  {label}")
    print(f"    Accuracy:   {acc:.4f}")
    print(f"    Macro-F1:   {float(prf['macro_f1']):.4f}")
    print(f"    Macro-Prec: {float(prf['macro_precision']):.4f}")
    print(f"    Macro-Rec:  {float(prf['macro_recall']):.4f}")
    print(f"    ROC-AUC:    {auc:.4f}")
    print(f"    Confusion matrix (rows=actual, cols=predicted):")
    print(f"      TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"      FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
 
    return {
        "label": label,
        "accuracy": round(acc, 6),
        "macro_f1": round(float(prf["macro_f1"]), 6),
        "macro_precision": round(float(prf["macro_precision"]), 6),
        "macro_recall": round(float(prf["macro_recall"]), 6),
        "auc": round(auc, 6),
        "TN": int(cm[0,0]), "FP": int(cm[0,1]),
        "FN": int(cm[1,0]), "TP": int(cm[1,1]),
    }
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",          type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir",        type=Path, default=Path("outputs"))
    parser.add_argument("--skip-fttransformer", action="store_true")
    args = parser.parse_args()
 
    # ── Load training data ────────────────────────────────────────────────────
    print("Loading training data …")
    df_train = pd.read_parquet(args.data_dir / "airbnb_features_train.parquet")
    df_train = filter_active_short_term(df_train)
 
    RAW = "target_unavailable_rate_90"
    threshold = float(df_train[RAW].quantile(0.75))
    df_train["target_high_demand"] = (df_train[RAW] >= threshold).astype(int)
    print(f"  Train: {len(df_train):,} rows  threshold={threshold:.4f}  "
          f"positive_rate={df_train['target_high_demand'].mean():.3f}")
 
    # ── Load test data ────────────────────────────────────────────────────────
    print("\nLoading temporal test data …")
    df_test = pd.read_parquet(args.data_dir / "airbnb_features_test.parquet")
    df_test = filter_active_short_term(df_test)
    df_test["target_high_demand"] = (df_test[RAW] >= threshold).astype(int)
    print(f"  Test:  {len(df_test):,} rows  "
          f"positive_rate={df_test['target_high_demand'].mean():.3f}")
 
    # ── Load generalisation data ──────────────────────────────────────────────
    print("\nLoading geographic generalisation data (Sydney, Tasmania, WA) …")
    df_gen = pd.read_parquet(args.data_dir / "airbnb_features_generalization.parquet")
    df_gen = filter_active_short_term(df_gen)
    df_gen["target_high_demand"] = (df_gen[RAW] >= threshold).astype(int)
    print(f"  Gen:   {len(df_gen):,} rows  "
          f"positive_rate={df_gen['target_high_demand'].mean():.3f}")
 
    with open(args.data_dir / "airbnb_feature_groups.json") as f:
        groups = json.load(f)
 
    base = select_numeric(df_train, groups.get("base_controls", []))
    text = (select_numeric(df_train, groups.get("host_nlp", groups.get("host_all", []))) +
            select_numeric(df_train, groups.get("review_nlp", groups.get("review_all", []))) +
            select_numeric(df_train, groups.get("description_nlp", [])))
 
    feat_sets = {"base_only": base, "base_plus_text": base + text}
    y_train = df_train["target_high_demand"].to_numpy()
    y_test  = df_test["target_high_demand"].to_numpy()
    y_gen   = df_gen["target_high_demand"].to_numpy()
 
    all_records = []
 
    # ── Evaluate each model × feature set ─────────────────────────────────────
    models_to_run = [
        ("Logistic",  lambda: LogisticModel(**BEST_PARAMS["Logistic"])),
        ("CatBoost",  lambda: CatBoostModel(**BEST_PARAMS["CatBoost"])),
    ]
    if not args.skip_fttransformer:
        models_to_run.append(
            ("FTTransformer", lambda: FTTransformerModel(**BEST_PARAMS["FTTransformer"]))
        )
 
    print(f"\n{'═'*60}")
    print("  FINAL TEST SET EVALUATION")
    print(f"{'═'*60}")
    print("  (Threshold from training set — no information from test used)")
 
    for feat_name, feat_cols in feat_sets.items():
        X_tr = df_train[feat_cols].to_numpy()
        X_te = df_test[feat_cols].to_numpy()
        X_ge = df_gen[feat_cols].to_numpy()
 
        for model_name, model_fn in models_to_run:
            print(f"\n{'─'*60}")
            print(f"  {model_name} | {feat_name}")
            print(f"{'─'*60}")
 
            # Impute — fit on train only
            imp = _MedianImputer()
            X_tr_imp = imp.fit_transform(X_tr)
            X_te_imp  = imp.transform(X_te)
            X_ge_imp  = imp.transform(X_ge)
 
            # Fit on full training set with best CV hyperparameters
            model = model_fn()
            model.fit(X_tr_imp, y_train)
 
            # Temporal test set
            rec = evaluate(model, X_te_imp, y_test,
                           f"Temporal test ({len(df_test):,} listings, latest snapshot)")
            rec.update({"model": model_name, "features": feat_name, "split": "temporal_test"})
            all_records.append(rec)
 
            # Generalisation test (CatBoost base_only only — best model)
            if model_name == "CatBoost" and feat_name == "base_only":
                rec2 = evaluate(model, X_ge_imp, y_gen,
                                f"Geographic generalisation (Sydney/Tasmania/WA, {len(df_gen):,} listings)")
                rec2.update({"model": model_name, "features": feat_name, "split": "generalisation"})
                all_records.append(rec2)
 
    out_dir = args.output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_set_results.csv"
    pd.DataFrame(all_records).to_csv(out_path, index=False)
 
    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    for r in all_records:
        print(f"  {r['model']:14s} {r['features']:15s} [{r['split']:18s}]  "
              f"acc={r['accuracy']:.3f}  f1={r['macro_f1']:.3f}  auc={r['auc']:.3f}")
 
    print(f"\nSaved → {out_path}")
 
 
if __name__ == "__main__":
    main()