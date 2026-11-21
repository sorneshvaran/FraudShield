"""
FraudShield - retraining with reviewed transactions
===================================================

Mirrors modules/model.py so a retrained model behaves like the original:

* same preprocessing  (median imputer + missing-value indicators)
* same estimator family and hyper-parameters as the ACTIVE model
  (XGBoost or Random Forest)
* same chronological 75 / 25 train / validation split of the original
  "earlier" data
* same threshold rule (best F1 on validation)

What is different
* every transaction you REVIEWED (confirmed fraud / confirmed legit) is added.
  Unreviewed transactions are never used: the model's own guesses are not
  ground truth and training on them would only reinforce its mistakes.
* reviewed rows can be up-weighted (new_row_weight) because there are few.
* a stratified slice (25 %) of the reviewed rows is held out so we can see
  how old and new models do on data they have not trained on.
* the original test set stays a fixed holdout. The new model is only
  RECOMMENDED for activation if it does not regress there.

The candidate is saved to models/versions/<version>.pkl; activating it is a
separate step handled by the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "fraud_intelligence"
TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"
VERSIONS_DIR = BASE_DIR / "models" / "versions"

Progress = Callable[[str, float], None]


class RetrainError(Exception):
    pass


# ------------------------------------------------------------------ helpers


def prepare_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Same conversion as modules/model.py::prepare_features."""
    X = df[features].copy()
    for column in features:
        if X[column].dtype == "object" or str(X[column].dtype).startswith("string"):
            X[column] = X[column].replace(
                {"True": 1, "False": 0, "true": 1, "false": 0}
            )
        X[column] = pd.to_numeric(X[column], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def make_estimator(kind: str, y: pd.Series, w: np.ndarray):
    if "RandomForest" in kind:
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    neg = float(w[y.to_numpy() == 0].sum())
    pos = float(w[y.to_numpy() == 1].sum())
    return XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=neg / max(pos, 1e-9),
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )


def evaluate(pipeline: Pipeline, threshold: float, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = y.nunique() == 2
    return {
        "rows": int(len(y)),
        "fraud_rows": int(y.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)) if both else None,
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "threshold": float(threshold),
    }


def _new_rows_frame(labeled: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    rows = []
    for r in labeled:
        row = {f: r["features"].get(f) for f in features}
        row["Class"] = int(r["label"])
        row["transaction_id"] = r["transaction_id"]
        rows.append(row)
    df = pd.DataFrame(rows)
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


# --------------------------------------------------------------------- main


def retrain(
    active_package: dict[str, Any],
    labeled: list[dict[str, Any]],
    version: str,
    new_row_weight: float = 3.0,
    progress: Optional[Progress] = None,
) -> dict[str, Any]:
    def step(msg: str, pct: float) -> None:
        if progress:
            progress(msg, pct)

    if not labeled:
        raise RetrainError("There are no reviewed transactions to train on.")
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            raise RetrainError(f"Original training data not found: {path}")

    old_pipeline: Pipeline = active_package["model"]
    features: list[str] = list(active_package["features"])
    old_threshold = float(active_package["threshold"])
    kind = type(old_pipeline.named_steps["classifier"]).__name__

    # ---- original data -------------------------------------------------
    step("Loading original training and test data", 0.05)
    earlier = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    required = features + ["Class", "timestamp"]
    for name, df in (("train_data.csv", earlier), ("test_data.csv", test)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RetrainError(f"{name} is missing columns: {missing}")

    earlier["timestamp"] = pd.to_datetime(earlier["timestamp"])
    earlier = earlier.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    cutoff = earlier.loc[int(len(earlier) * 0.75), "timestamp"]
    base_train = earlier[earlier["timestamp"] < cutoff]
    base_val = earlier[earlier["timestamp"] >= cutoff]

    X_base_train = prepare_features(base_train, features)
    y_base_train = base_train["Class"].astype(int)
    X_base_val = prepare_features(base_val, features)
    y_base_val = base_val["Class"].astype(int)
    X_test = prepare_features(test, features)
    y_test = test["Class"].astype(int)

    # ---- reviewed rows -------------------------------------------------
    step(f"Preparing {len(labeled)} reviewed transactions", 0.15)
    new_df = _new_rows_frame(labeled, features)
    X_new = new_df[features]
    y_new = new_df["Class"].astype(int)

    holdout_idx: list[int] = []
    train_idx: list[int] = list(range(len(new_df)))
    if len(new_df) >= 12:
        try:
            counts = y_new.value_counts()
            strat = y_new if (len(counts) == 2 and counts.min() >= 2) else None
            train_idx, holdout_idx = (
                list(a) for a in train_test_split(
                    np.arange(len(new_df)), test_size=0.25,
                    random_state=42, stratify=strat,
                )
            )
        except ValueError:
            train_idx, holdout_idx = list(range(len(new_df))), []

    X_new_train, y_new_train = X_new.iloc[train_idx], y_new.iloc[train_idx]
    X_new_hold, y_new_hold = X_new.iloc[holdout_idx], y_new.iloc[holdout_idx]

    # ---- fit -----------------------------------------------------------
    X_train = pd.concat([X_base_train, X_new_train], ignore_index=True)
    y_train = pd.concat([y_base_train, y_new_train], ignore_index=True)
    weights = np.concatenate([
        np.ones(len(X_base_train)),
        np.full(len(X_new_train), float(new_row_weight)),
    ])

    step(f"Training {kind} on {len(X_train):,} rows "
         f"({len(X_new_train)} reviewed, weight x{new_row_weight:g})", 0.30)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", make_estimator(kind, y_train, weights)),
    ])
    pipeline.fit(X_train, y_train, classifier__sample_weight=weights)

    # ---- threshold on validation (+ held-out reviewed rows) -------------
    step("Choosing the decision threshold on validation data", 0.70)
    X_val = pd.concat([X_base_val, X_new_hold], ignore_index=True)
    y_val = pd.concat([y_base_val, y_new_hold], ignore_index=True)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_proba)
    val_ap = float(average_precision_score(y_val, val_proba))

    # ---- compare with the active model ----------------------------------
    step("Comparing with the active model on the original test set", 0.82)
    metrics: dict[str, Any] = {
        "new_on_test": evaluate(pipeline, threshold, X_test, y_test),
        "old_on_test": evaluate(old_pipeline, old_threshold, X_test, y_test),
    }
    if len(X_new_hold):
        metrics["new_on_new_holdout"] = evaluate(pipeline, threshold, X_new_hold, y_new_hold)
        metrics["old_on_new_holdout"] = evaluate(old_pipeline, old_threshold, X_new_hold, y_new_hold)

    new_t, old_t = metrics["new_on_test"], metrics["old_on_test"]
    problems = []
    if (new_t["pr_auc"] or 0) < (old_t["pr_auc"] or 0) - 0.005:
        problems.append("PR-AUC on the original test set dropped by more than 0.005")
    if new_t["recall"] < old_t["recall"] - 0.02:
        problems.append("recall on the original test set dropped by more than 2 points")
    if new_t["precision"] < old_t["precision"] - 0.03:
        problems.append("precision on the original test set dropped by more than 3 points")
    recommended = not problems
    reason = (
        "No regression on the original test set." if recommended
        else "Not recommended: " + "; ".join(problems) + "."
    )

    # ---- save candidate -------------------------------------------------
    step("Saving the candidate model", 0.93)
    base_name = str(active_package.get("model_name", kind)).split(" (retrained")[0]
    model_name = f"{base_name} (retrained {version})"
    package = {
        "model": pipeline,
        "features": features,
        "model_name": model_name,
        "threshold": threshold,
        "validation_average_precision": val_ap,
    }
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"{version}.pkl"
    joblib.dump(package, path)

    step("Done", 1.0)
    return {
        "version": version,
        "path": str(path),
        "model_name": model_name,
        "threshold": threshold,
        "validation_average_precision": val_ap,
        "n_original": int(len(earlier)),
        "n_new": int(len(new_df)),
        "n_new_train": int(len(X_new_train)),
        "n_new_holdout": int(len(X_new_hold)),
        "new_row_weight": float(new_row_weight),
        "metrics": metrics,
        "recommended": recommended,
        "recommendation_reason": reason,
        "used_transaction_ids": new_df["transaction_id"].tolist(),
    }