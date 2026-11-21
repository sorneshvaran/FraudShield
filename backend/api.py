"""
FraudShield - FastAPI backend
=============================

Run from the FraudShield/ folder:

    uvicorn backend.api:app --port 8000

Interactive docs:  http://127.0.0.1:8000/docs

What it does
------------
* Scores a transaction you type in (no payment gateway needed): rebuilds the
  same engineered features used in training, predicts, explains with SHAP and
  returns a risk level + recommended action
* Saves transactions to SQLite (data/fraudshield.db) so history survives restarts
* Lets you REVIEW saved transactions (confirmed fraud / confirmed legit)
* Writes an analyst briefing with a Groq LLM (your own key)
* Retrains the model from the original data + your reviewed transactions,
  compares it with the active model, and lets you activate / roll back versions

Optional environment variables (put them in FraudShield/.env):
    GROQ_API_KEY          your Groq key (or paste it in the dashboard sidebar)
    GROQ_MODEL            default Groq model id
    FRAUDSHIELD_API_KEY   if set, every request must send header  X-API-Key
    FRAUDSHIELD_CORS      comma separated origins, default "*"
    FRAUDSHIELD_DB        custom SQLite path
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import shutil
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend import db, llm
from backend import retrain as retrain_mod

try:  # SHAP is optional at runtime: the API still scores without it.
    import shap
except Exception:  # pragma: no cover
    shap = None

# =====================================================
# PATHS / CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
VERSIONS_DIR = retrain_mod.VERSIONS_DIR
HISTORY_PATH = BASE_DIR / "data" / "processed_data.csv"
REPORT_DIR = BASE_DIR / "data" / "fraud_intelligence" / "model_reports"
EXPLAIN_DIR = BASE_DIR / "data" / "explainability"

API_KEY = os.getenv("FRAUDSHIELD_API_KEY", "").strip()
CORS_ORIGINS = [
    o.strip() for o in os.getenv("FRAUDSHIELD_CORS", "*").split(",") if o.strip()
]

MAX_BATCH = 2000
SUPPORTED_CURRENCIES = {"INR", "USD", "EUR", "GBP", "AED", "SGD"}

logger = logging.getLogger("fraudshield")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =====================================================
# SCHEMAS
# =====================================================


class TransactionIn(BaseModel):
    """A transaction to score. Customer profile fields are looked up
    automatically for known customers."""

    customer_id: str
    amount: float = Field(..., gt=0, description="Amount in the transaction currency")
    currency: str = "INR"
    timestamp: Optional[datetime] = Field(
        None, description="Defaults to the current server time"
    )

    merchant_id: Optional[str] = None
    payee_id: Optional[str] = None
    device_id: Optional[str] = None
    merchant_category: Optional[str] = None
    transaction_type: Optional[str] = Field(
        None, description="UPI, Debit Card, Credit Card, NetBanking, Wallet"
    )
    transaction_subtype: Optional[str] = None
    transaction_status: str = "SUCCESS"

    card_present: int = 0
    is_international: int = 0
    transaction_latitude: float = Field(..., ge=-90, le=90)
    transaction_longitude: float = Field(..., ge=-180, le=180)

    # Only needed for customers that are not in the history file.
    customer_registered_latitude: Optional[float] = Field(None, ge=-90, le=90)
    customer_registered_longitude: Optional[float] = Field(None, ge=-180, le=180)
    account_creation_date: Optional[datetime] = None

    persist: bool = Field(
        True,
        description="Save to the SQLite database and to the customer's history, "
        "so later checks see it (velocity, new-payee, spending baseline).",
    )
    review_label: Optional[int] = Field(
        None,
        description="Optional ground truth (1 = fraud, 0 = legit). Only used when the "
        "transaction is saved; it is stored as the review label used for retraining.",
    )

    @field_validator("review_label")
    @classmethod
    def _review_label(cls, v: Optional[int]) -> Optional[int]:
        if v not in (None, 0, 1):
            raise ValueError("review_label must be 0, 1 or null")
        return v

    @field_validator("currency")
    @classmethod
    def _currency(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency. Use one of {sorted(SUPPORTED_CURRENCIES)}")
        return v

    @field_validator("transaction_status")
    @classmethod
    def _status(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in {"SUCCESS", "FAILED"}:
            raise ValueError("transaction_status must be SUCCESS or FAILED")
        return v

    @field_validator("card_present", "is_international", mode="before")
    @classmethod
    def _bool_int(cls, v: Any) -> int:
        if isinstance(v, str):
            return 1 if v.strip().lower() in {"1", "true", "yes", "y"} else 0
        return int(bool(v))


class Reason(BaseModel):
    feature: str
    label: str
    value: Optional[float | str] = None
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"
    text: str


class PredictionOut(BaseModel):
    transaction_id: str
    customer_id: str
    fraud_probability: float
    threshold: float
    is_fraud: bool
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    recommended_action: str
    reasons: list[Reason]
    features: dict[str, Optional[float]]
    base_value: Optional[float] = None
    explanation_unit: Optional[str] = None
    model_name: str
    model_version: Optional[str] = None
    scored_at: datetime
    saved: bool = False
    ai_explanation: Optional[str] = None
    ai_model: Optional[str] = None
    ai_error: Optional[str] = None


class BatchIn(BaseModel):
    transactions: list[TransactionIn]
    persist: bool = False


class BatchOut(BaseModel):
    count: int
    flagged: int
    results: list[PredictionOut]
    errors: list[dict[str, Any]]


class ReviewIn(BaseModel):
    label: Optional[int] = Field(
        None, description="1 = confirmed fraud, 0 = confirmed legit, null = clear the review"
    )
    note: Optional[str] = Field(None, max_length=500)

    @field_validator("label")
    @classmethod
    def _label(cls, v: Optional[int]) -> Optional[int]:
        if v not in (None, 0, 1):
            raise ValueError("label must be 0, 1 or null")
        return v


class RetrainIn(BaseModel):
    new_row_weight: float = Field(3.0, ge=1, le=20, description="How many training rows each reviewed transaction counts as")
    min_new_rows: int = Field(10, ge=1, description="Refuse to retrain with fewer reviewed transactions")
    auto_activate: bool = Field(False, description="Activate the new model if it shows no regression")


# =====================================================
# FEATURE LABELS / REASON TEXT
# =====================================================

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "card_present": "Card present",
    "is_international": "International transaction",
    "transaction_hour": "Hour of day",
    "transaction_day": "Day of week",
    "is_weekend": "Weekend",
    "account_age_days": "Account age (days)",
    "distance_from_home_km": "Distance from home (km)",
    "device_change": "Device changed",
    "transactions_last_1h": "Transactions in last hour",
    "transactions_last_24h": "Transactions in last 24h",
    "failed_attempts_24h": "Failed attempts in last 24h",
    "customer_prior_transaction_count": "Prior transactions",
    "customer_avg_amount_prior": "Customer's average amount",
    "customer_std_amount_prior": "Customer's amount std-dev",
    "amount_deviation_ratio": "Amount vs. customer average",
    "unusual_transaction_time": "Unusual time for this customer",
    "new_merchant_for_customer": "New merchant",
    "new_payee": "New payee",
    "has_time_history": "Has time history",
    "has_merchant_history": "Has merchant history",
    "has_payee_history": "Has payee history",
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def describe_reason(feature: str, value: Optional[float], currency: str) -> str:
    """Plain-language sentence for a feature value."""
    v = value
    if v is None:
        return "Value not available for this customer (history too short)."
    if feature == "amount":
        return f"Amount is {v:,.2f} {currency}."
    if feature == "amount_deviation_ratio":
        return f"Amount is {v:.1f}x this customer's usual spend."
    if feature == "distance_from_home_km":
        return f"Transaction happened {v:,.0f} km from the registered home location."
    if feature == "new_payee":
        return "Money is going to a payee this customer has never paid before." if v else "Payee is known to this customer."
    if feature == "new_merchant_for_customer":
        return "First time this customer shops at this merchant." if v else "Merchant is known to this customer."
    if feature == "device_change":
        return "A different device than the customer's previous transaction." if v else "Same device as the previous transaction."
    if feature == "is_international":
        return "International transaction." if v else "Domestic transaction."
    if feature == "card_present":
        return "Card was physically present." if v else "Card-not-present transaction."
    if feature == "transaction_hour":
        return f"Transaction at {int(v):02d}:00."
    if feature == "transaction_day":
        return f"Transaction on a {DAYS[int(v) % 7]}."
    if feature == "unusual_transaction_time":
        return "Outside this customer's usual hours." if v else "Within this customer's usual hours."
    if feature == "account_age_days":
        return f"Account is {int(v)} days old."
    if feature == "transactions_last_1h":
        return f"{int(v)} earlier transaction(s) in the last hour."
    if feature == "transactions_last_24h":
        return f"{int(v)} earlier transaction(s) in the last 24 hours."
    if feature == "failed_attempts_24h":
        return f"{int(v)} failed attempt(s) in the last 24 hours."
    if feature == "customer_prior_transaction_count":
        return f"Customer has {int(v)} prior transaction(s) on record."
    if feature == "customer_avg_amount_prior":
        return f"Customer's average prior amount is {v:,.2f} {currency}."
    if feature == "customer_std_amount_prior":
        return f"Customer's spending std-dev is {v:,.2f} {currency}."
    if feature == "is_weekend":
        return "Weekend transaction." if v else "Weekday transaction."
    return f"{FEATURE_LABELS.get(feature, feature)} = {v:g}."


def risk_level(prob: float, threshold: float) -> tuple[str, str]:
    if prob < threshold * 0.5:
        return "LOW", "Approve"
    if prob < threshold:
        return "MEDIUM", "Approve with step-up authentication (OTP / biometric)"
    if prob < 0.85:
        return "HIGH", "Hold transaction and send for analyst review"
    return "CRITICAL", "Block transaction and contact the customer"


# =====================================================
# HISTORY STORE  (customer history + profiles)
# =====================================================

HISTORY_COLS = [
    "timestamp", "amount", "currency", "transaction_status",
    "device_id", "merchant_id", "payee_id", "transaction_hour",
]


def _db_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["transaction_hour"] = df["timestamp"].dt.hour
    return df


class HistoryStore:
    """In-memory customer history used to rebuild behavioural features.

    history[customer] = original CSV rows (base) + saved SQLite rows.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.base: dict[str, pd.DataFrame] = {}
        self.history: dict[str, pd.DataFrame] = {}
        self.profiles: dict[str, dict[str, Any]] = {}
        self.csv_customers: set[str] = set()
        self.raw: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------ loading
    def load(self, path: Path) -> None:
        if path.exists():
            self._load_csv(path)
        else:
            logger.warning("History file not found: %s", path)
        self._load_db()
        logger.info("History ready: %d customers (%d from the database)",
                    len(self.profiles), len(self.profiles) - len(self.csv_customers))

    def _load_csv(self, path: Path) -> None:
        df = pd.read_csv(path, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["account_creation_date"] = pd.to_datetime(df["account_creation_date"], errors="coerce")
        df = df.dropna(subset=["timestamp", "customer_id"]).copy()
        df["customer_id"] = df["customer_id"].astype(str)
        df["transaction_hour"] = df["timestamp"].dt.hour
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        self.raw = df

        first = df.drop_duplicates("customer_id", keep="first")
        for row in first.itertuples(index=False):
            self.profiles[row.customer_id] = {
                "customer_id": row.customer_id,
                "registered_latitude": _num(row.customer_registered_latitude),
                "registered_longitude": _num(row.customer_registered_longitude),
                "account_creation_date": (
                    row.account_creation_date.to_pydatetime()
                    if pd.notna(row.account_creation_date) else None
                ),
            }
        self.csv_customers = set(self.profiles)

        for cid, g in df.groupby("customer_id", sort=False):
            frame = g[HISTORY_COLS].reset_index(drop=True)
            self.base[cid] = frame
            self.history[cid] = frame

    def _load_db(self) -> None:
        for c in db.list_customers():
            cid = c["customer_id"]
            if cid not in self.profiles:
                created = pd.to_datetime(c["account_creation_date"], errors="coerce")
                self.profiles[cid] = {
                    "customer_id": cid,
                    "registered_latitude": c["registered_latitude"],
                    "registered_longitude": c["registered_longitude"],
                    "account_creation_date": None if pd.isna(created) else created.to_pydatetime(),
                }
        rows = db.history_rows()
        if not rows:
            return
        df = _db_frame(rows)
        for cid, g in df.groupby("customer_id", sort=False):
            self.history[str(cid)] = self._merge(self.base.get(str(cid)), g[HISTORY_COLS])

    @staticmethod
    def _merge(base: Optional[pd.DataFrame], extra: pd.DataFrame) -> pd.DataFrame:
        parts = [p for p in (base, extra) if p is not None and not p.empty]
        if not parts:
            return pd.DataFrame(columns=HISTORY_COLS)
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------ access
    def get(self, customer_id: str) -> pd.DataFrame:
        with self.lock:
            h = self.history.get(customer_id)
            return h.copy() if h is not None else pd.DataFrame(columns=HISTORY_COLS)

    def append(self, customer_id: str, row: dict[str, Any], profile: Optional[dict] = None) -> None:
        with self.lock:
            new = pd.DataFrame([row], columns=HISTORY_COLS)
            self.history[customer_id] = self._merge(self.history.get(customer_id), new)
            if profile and customer_id not in self.profiles:
                self.profiles[customer_id] = profile

    def rebuild(self, customer_id: str) -> None:
        """Recompute one customer's history from CSV + database (after a delete)."""
        with self.lock:
            rows = db.history_rows(customer_id)
            extra = _db_frame(rows)[HISTORY_COLS] if rows else None
            merged = self._merge(self.base.get(customer_id), extra if extra is not None else pd.DataFrame(columns=HISTORY_COLS))
            self.history[customer_id] = merged


store = HistoryStore()


# =====================================================
# FEATURE ENGINEERING (mirror of modules/feature_engineering.py)
# =====================================================


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def resolve_profile(txn: TransactionIn) -> dict[str, Any]:
    profile = store.profiles.get(txn.customer_id)
    if profile:
        return profile

    if (
        txn.customer_registered_latitude is None
        or txn.customer_registered_longitude is None
        or txn.account_creation_date is None
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown customer '{txn.customer_id}'. Send customer_registered_latitude, "
                "customer_registered_longitude and account_creation_date to score a new customer."
            ),
        )
    return {
        "customer_id": txn.customer_id,
        "registered_latitude": txn.customer_registered_latitude,
        "registered_longitude": txn.customer_registered_longitude,
        "account_creation_date": txn.account_creation_date,
    }


def build_features(
    txn: TransactionIn, ts: datetime, profile: dict[str, Any], hist: pd.DataFrame
) -> dict[str, Optional[float]]:
    """Recreate every model feature using ONLY strictly earlier history,
    exactly like the training pipeline does."""

    prior = hist[hist["timestamp"] < ts] if not hist.empty else hist
    hour = ts.hour
    day = ts.weekday()

    f: dict[str, Optional[float]] = {
        "amount": float(txn.amount),
        "card_present": int(txn.card_present),
        "is_international": int(txn.is_international),
        "transaction_hour": hour,
        "transaction_day": day,
        "is_weekend": int(day in (5, 6)),
    }

    created = profile["account_creation_date"]
    f["account_age_days"] = float((ts - pd.Timestamp(created)).days) if created else None

    f["distance_from_home_km"] = haversine_km(
        profile["registered_latitude"], profile["registered_longitude"],
        txn.transaction_latitude, txn.transaction_longitude,
    )

    # --- device change: compare with the latest earlier transaction
    device_change = 0
    if not prior.empty and txn.device_id:
        last = prior.iloc[-1]
        if pd.notna(last["device_id"]) and last["device_id"] != txn.device_id:
            device_change = 1
    f["device_change"] = device_change

    # --- velocity
    if prior.empty:
        f["transactions_last_1h"] = 0
        f["transactions_last_24h"] = 0
        f["failed_attempts_24h"] = 0
    else:
        t1 = ts - pd.Timedelta(hours=1)
        t24 = ts - pd.Timedelta(hours=24)
        f["transactions_last_1h"] = int((prior["timestamp"] >= t1).sum())
        last24 = prior[prior["timestamp"] >= t24]
        f["transactions_last_24h"] = int(len(last24))
        f["failed_attempts_24h"] = int((last24["transaction_status"] == "FAILED").sum())

    # --- spending baseline (same currency, successful, positive amounts)
    base = prior[
        (prior["currency"] == txn.currency)
        & (prior["transaction_status"] == "SUCCESS")
        & (prior["amount"] > 0)
    ] if not prior.empty else prior
    n = len(base)
    f["customer_prior_transaction_count"] = n
    if n > 0:
        mean = float(base["amount"].mean())
        f["customer_avg_amount_prior"] = mean
        f["customer_std_amount_prior"] = float(base["amount"].std(ddof=1)) if n >= 2 else None
        f["amount_deviation_ratio"] = float(txn.amount) / mean if mean > 0 else None
    else:
        f["customer_avg_amount_prior"] = None
        f["customer_std_amount_prior"] = None
        f["amount_deviation_ratio"] = None

    # --- time habits: >4 circular hours from every earlier transaction hour
    if prior.empty:
        f["unusual_transaction_time"] = 0
        f["has_time_history"] = 0
    else:
        hours = prior["transaction_hour"].to_numpy()
        diff = np.abs(hour - hours)
        circ = np.minimum(diff, 24 - diff)
        f["unusual_transaction_time"] = int(circ.min() > 4)
        f["has_time_history"] = 1

    # --- merchant / payee novelty
    seen_m = set(prior["merchant_id"].dropna()) if not prior.empty else set()
    seen_p = set(prior["payee_id"].dropna()) if not prior.empty else set()

    if txn.merchant_id:
        f["has_merchant_history"] = int(len(seen_m) > 0)
        f["new_merchant_for_customer"] = int(txn.merchant_id not in seen_m)
    else:
        f["has_merchant_history"] = 0
        f["new_merchant_for_customer"] = 0

    if txn.payee_id:
        f["has_payee_history"] = int(len(seen_p) > 0)
        f["new_payee"] = int(txn.payee_id not in seen_p)
    else:
        f["has_payee_history"] = 0
        f["new_payee"] = 0

    return f


# =====================================================
# MODEL SERVICE  (hot-swappable)
# =====================================================


@dataclass
class ModelState:
    pipeline: Any
    features: list[str]
    threshold: float
    name: str
    version: str
    explainer: Any
    names: list[str]


class ModelService:
    """Holds the active model. The whole state is swapped in ONE assignment,
    so a request never sees a half-updated model."""

    def __init__(self) -> None:
        self.state: Optional[ModelState] = None
        self.loaded_at: Optional[datetime] = None

    # --- convenience accessors (used by the endpoints)
    @property
    def pipeline(self):
        return self.state.pipeline if self.state else None

    @property
    def features(self) -> list[str]:
        return self.state.features if self.state else []

    @property
    def threshold(self) -> float:
        return self.state.threshold if self.state else 0.5

    @property
    def model_name(self) -> str:
        return self.state.name if self.state else "unknown"

    @property
    def version(self) -> Optional[str]:
        return self.state.version if self.state else None

    @property
    def explainer(self):
        return self.state.explainer if self.state else None

    def package(self) -> dict[str, Any]:
        s = self.state
        return {
            "model": s.pipeline, "features": s.features,
            "threshold": s.threshold, "model_name": s.name,
        }

    # --- loading
    def load(self, path: Path, version: str = "v0") -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Train it first:  python modules/model.py"
            )
        self.set_package(joblib.load(path), version)

    def set_package(self, package: dict[str, Any], version: str) -> None:
        pipeline = package["model"]
        features = list(package["features"])
        explainer, names = None, []

        if shap is not None:
            try:
                explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
                imputer = pipeline.named_steps["imputer"]
                try:
                    names = imputer.get_feature_names_out(np.array(features)).tolist()
                except Exception:
                    names = list(features)
                    indicator = getattr(imputer, "indicator_", None)
                    if indicator is not None:
                        names += [f"missingindicator_{features[i]}" for i in indicator.features_]
            except Exception as exc:  # pragma: no cover
                logger.warning("SHAP explainer unavailable: %s", exc)
                explainer, names = None, []

        self.state = ModelState(
            pipeline=pipeline,
            features=features,
            threshold=float(package["threshold"]),
            name=str(package.get("model_name", "model")),
            version=version,
            explainer=explainer,
            names=names,
        )
        self.loaded_at = datetime.now()
        logger.info("Model active: %s [%s] | threshold %.4f | %d features | SHAP %s",
                    self.state.name, version, self.state.threshold, len(features),
                    "on" if explainer is not None else "off")

    # --- inference
    @staticmethod
    def _frame(s: ModelState, feats: dict[str, Optional[float]]) -> pd.DataFrame:
        row = {c: (np.nan if feats.get(c) is None else float(feats[c])) for c in s.features}
        return pd.DataFrame([row], columns=s.features)

    def predict(self, feats: dict[str, Optional[float]]) -> tuple[float, ModelState]:
        s = self.state
        return float(s.pipeline.predict_proba(self._frame(s, feats))[0, 1]), s

    def explain(
        self, s: ModelState, feats: dict[str, Optional[float]], currency: str, top_n: int = 6
    ) -> tuple[list[Reason], Optional[float], Optional[str]]:
        if s.explainer is None:
            return [], None, None
        try:
            Xt = pd.DataFrame(
                s.pipeline.named_steps["imputer"].transform(self._frame(s, feats)),
                columns=s.names,
            )
            exp = s.explainer(Xt)
            vals = np.asarray(exp.values)
            vals = vals[0, :, 1] if vals.ndim == 3 else vals[0]
            base = np.asarray(exp.base_values).reshape(-1)
            base_value = float(base[-1]) if base.size else None
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)
            return [], None, None

        kind = type(s.pipeline.named_steps["classifier"]).__name__.upper()
        unit = "log-odds" if "XGB" in kind else "probability"

        reasons: list[Reason] = []
        for i in np.argsort(-np.abs(vals)):
            name = s.names[i]
            if name.startswith("missingindicator_") or abs(vals[i]) < 1e-9:
                continue
            raw = feats.get(name)
            reasons.append(
                Reason(
                    feature=name,
                    label=FEATURE_LABELS.get(name, name),
                    value=raw,
                    shap_value=float(vals[i]),
                    direction="increases_risk" if vals[i] > 0 else "decreases_risk",
                    text=describe_reason(name, raw, currency),
                )
            )
            if len(reasons) >= top_n:
                break
        return reasons, base_value, unit


model = ModelService()


# =====================================================
# MODEL VERSION REGISTRY
# =====================================================


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def bootstrap_versions() -> None:
    """Load the active model and make sure the registry knows about it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first:  python modules/model.py"
        )
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    versions = db.list_versions()
    active = next((v for v in versions if v["is_active"]), None)

    # Registry empty -> register the model that is currently on disk as v0.
    # Registry present but the file on disk was replaced behind our back
    # (e.g. you re-ran modules/model.py) -> register it as a new version.
    if active is None or not Path(active["path"]).exists() or _sha1(Path(active["path"])) != _sha1(MODEL_PATH):
        number = db.next_version_number()
        version = f"v{number - 1}" if not versions else f"v{number}"
        source = "original" if not versions else "external"
        dest = VERSIONS_DIR / f"{version}.pkl"
        shutil.copyfile(MODEL_PATH, dest)
        pkg = joblib.load(dest)
        metrics = {}
        if source == "original":
            tm = (_read_csv_records(REPORT_DIR / "final_test_metrics.csv") or [None])[0]
            if tm:
                metrics = {"test_report": tm}
        db.insert_version({
            "version": version, "source": source,
            "model_name": str(pkg.get("model_name", "model")),
            "path": str(dest), "threshold": float(pkg["threshold"]),
            "n_original": None, "n_new": 0, "new_row_weight": None,
            "metrics_json": metrics,
            "notes": "Model found on disk at startup." if source == "original"
                     else "models/fraud_model.pkl was replaced outside the API.",
            "is_active": 0,
        })
        db.set_active_version(version)
        active_version = version
    else:
        active_version = active["version"]

    model.load(MODEL_PATH, active_version)


def activate_version(version: str) -> dict[str, Any]:
    rec = db.get_version(version)
    if not rec:
        raise HTTPException(404, f"Unknown model version '{version}'")
    path = Path(rec["path"])
    if not path.exists():
        raise HTTPException(410, f"Model file for {version} is missing: {path}")
    pkg = joblib.load(path)
    model.set_package(pkg, version)  # swap first: if loading fails nothing changed on disk
    tmp = MODEL_PATH.with_suffix(".tmp")
    shutil.copyfile(path, tmp)
    os.replace(tmp, MODEL_PATH)
    db.set_active_version(version)
    return db.get_version(version)  # type: ignore[return-value]


# =====================================================
# RECENT PREDICTIONS (in memory)
# =====================================================

recent_predictions: deque[dict[str, Any]] = deque(maxlen=1000)
log_lock = threading.Lock()
_txn_counter = {"n": 0}


def _next_txn_id() -> str:
    with log_lock:
        _txn_counter["n"] += 1
        return f"TXN{datetime.now():%y%m%d%H%M%S}{_txn_counter['n']:04d}{secrets.token_hex(2).upper()}"


def _log_prediction(out: PredictionOut, txn: TransactionIn) -> None:
    with log_lock:
        recent_predictions.append({
            "transaction_id": out.transaction_id,
            "scored_at": out.scored_at.isoformat(timespec="seconds"),
            "customer_id": out.customer_id,
            "amount": txn.amount,
            "currency": txn.currency,
            "transaction_type": txn.transaction_type,
            "fraud_probability": round(out.fraud_probability, 6),
            "is_fraud": out.is_fraud,
            "risk_level": out.risk_level,
            "saved": False,
        })


# =====================================================
# SCORING / SAVING
# =====================================================


@dataclass
class Scored:
    out: PredictionOut
    row: dict[str, Any]       # history row for this transaction
    profile: dict[str, Any]
    ts: datetime


def score_transaction(
    txn: TransactionIn,
    *,
    explain: bool = True,
    hist_override: Optional[pd.DataFrame] = None,
) -> Scored:
    ts = pd.Timestamp(txn.timestamp or datetime.now()).to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)

    profile = resolve_profile(txn)
    hist = hist_override if hist_override is not None else store.get(txn.customer_id)

    feats = build_features(txn, ts, profile, hist)
    prob, state = model.predict(feats)
    level, action = risk_level(prob, state.threshold)
    reasons, base_value, unit = model.explain(state, feats, txn.currency) if explain else ([], None, None)

    out = PredictionOut(
        transaction_id=_next_txn_id(),
        customer_id=txn.customer_id,
        fraud_probability=prob,
        threshold=state.threshold,
        is_fraud=prob >= state.threshold,
        risk_level=level,
        recommended_action=action,
        reasons=reasons,
        features={k: (None if v is None else float(v)) for k, v in feats.items()},
        base_value=base_value,
        explanation_unit=unit,
        model_name=state.name,
        model_version=state.version,
        scored_at=datetime.now(),
    )
    row = {
        "timestamp": pd.Timestamp(ts),
        "amount": float(txn.amount),
        "currency": txn.currency,
        "transaction_status": txn.transaction_status,
        "device_id": txn.device_id,
        "merchant_id": txn.merchant_id,
        "payee_id": txn.payee_id,
        "transaction_hour": ts.hour,
    }
    _log_prediction(out, txn)
    return Scored(out=out, row=row, profile=profile, ts=ts)


def persist_scored(txn: TransactionIn, scored: Scored) -> None:
    """Save to SQLite first (source of truth), then to the in-memory history."""
    out = scored.out
    if txn.customer_id not in store.csv_customers:
        db.upsert_customer(scored.profile)

    db.insert_transaction({
        "transaction_id": out.transaction_id,
        "timestamp": scored.ts.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": txn.customer_id,
        "amount": float(txn.amount),
        "currency": txn.currency,
        "merchant_id": txn.merchant_id,
        "payee_id": txn.payee_id,
        "device_id": txn.device_id,
        "merchant_category": txn.merchant_category,
        "transaction_type": txn.transaction_type,
        "transaction_subtype": txn.transaction_subtype,
        "transaction_status": txn.transaction_status,
        "card_present": int(txn.card_present),
        "is_international": int(txn.is_international),
        "transaction_latitude": txn.transaction_latitude,
        "transaction_longitude": txn.transaction_longitude,
        "features_json": out.features,
        "reasons_json": [r.model_dump() for r in out.reasons],
        "fraud_probability": out.fraud_probability,
        "threshold": out.threshold,
        "predicted_fraud": int(out.is_fraud),
        "risk_level": out.risk_level,
        "recommended_action": out.recommended_action,
        "model_name": out.model_name,
        "model_version": out.model_version,
    })
    store.append(txn.customer_id, scored.row, scored.profile)
    out.saved = True
    if txn.review_label is not None:
        db.set_review(out.transaction_id, txn.review_label, "Label imported with the transaction")
    with log_lock:
        for rec in reversed(recent_predictions):
            if rec["transaction_id"] == out.transaction_id:
                rec["saved"] = True
                break


# =====================================================
# CUSTOMER SUMMARY (used by /customers and by the AI prompt)
# =====================================================


def _recent_unique(h: pd.DataFrame, col: str, n: int = 30) -> list[str]:
    """Most recently used distinct values of a column (newest first)."""
    if h.empty:
        return []
    vals = h[col].dropna().iloc[::-1].drop_duplicates().head(n)
    return [str(v) for v in vals]


def customer_summary(customer_id: str, recent: int = 0) -> Optional[dict[str, Any]]:
    profile = store.profiles.get(customer_id)
    if not profile:
        return None

    h = store.get(customer_id)
    ok = h[(h["transaction_status"] == "SUCCESS")] if not h.empty else h
    top_currency = h["currency"].mode().iat[0] if not h.empty else "INR"
    same = ok[ok["currency"] == top_currency] if not ok.empty else ok

    last_txns = []
    if recent and not h.empty:
        tail = h.tail(recent).iloc[::-1]
        last_txns = [
            {
                "timestamp": r.timestamp.isoformat(sep=" "),
                "amount": float(r.amount),
                "currency": r.currency,
                "status": r.transaction_status,
                "merchant_id": None if pd.isna(r.merchant_id) else r.merchant_id,
                "payee_id": None if pd.isna(r.payee_id) else r.payee_id,
                "device_id": None if pd.isna(r.device_id) else r.device_id,
            }
            for r in tail.itertuples(index=False)
        ]

    return {
        **profile,
        "transaction_count": int(len(h)),
        "failed_count": int((h["transaction_status"] == "FAILED").sum()) if not h.empty else 0,
        "primary_currency": top_currency,
        "avg_amount": float(same["amount"].mean()) if not same.empty else None,
        "max_amount": float(same["amount"].max()) if not same.empty else None,
        "known_merchants": int(h["merchant_id"].nunique()) if not h.empty else 0,
        "known_payees": int(h["payee_id"].nunique()) if not h.empty else 0,
        "last_device_id": (
            None if h.empty or pd.isna(h.iloc[-1]["device_id"]) else h.iloc[-1]["device_id"]
        ),
        "last_transaction_at": h.iloc[-1]["timestamp"].isoformat(sep=" ") if not h.empty else None,
        "usual_hours": sorted({int(x) for x in h["transaction_hour"]}) if not h.empty else [],
        "known_merchant_ids": _recent_unique(h, "merchant_id"),
        "known_payee_ids": _recent_unique(h, "payee_id"),
        "known_device_ids": _recent_unique(h, "device_id"),
        "recent_transactions": last_txns,
    }


# =====================================================
# AI (GROQ)
# =====================================================


def generate_ai(
    txn: dict[str, Any], result: dict[str, Any], groq_key: Optional[str], ai_model: Optional[str]
) -> tuple[str, str]:
    key = llm.resolve_key(groq_key)
    summary = customer_summary(txn["customer_id"])
    facts = llm.build_facts(txn, result, summary)
    return llm.explain(facts, key, ai_model)


def attach_ai(
    out: PredictionOut, txn: TransactionIn, groq_key: Optional[str], ai_model: Optional[str]
) -> None:
    """Never fails the prediction: problems are returned in out.ai_error."""
    try:
        text, used = generate_ai(
            txn.model_dump(mode="json"), out.model_dump(mode="json"), groq_key, ai_model
        )
        out.ai_explanation, out.ai_model = text, used
        if out.saved:
            db.set_ai(out.transaction_id, text, used)
    except llm.GroqError as exc:
        out.ai_error = exc.message
    except Exception as exc:  # pragma: no cover
        logger.exception("AI explanation failed")
        out.ai_error = f"AI explanation failed: {exc.__class__.__name__}"


# =====================================================
# RETRAIN JOB
# =====================================================

job_lock = threading.Lock()


def _fresh_job() -> dict[str, Any]:
    return {
        "state": "idle", "progress": 0.0, "message": "", "log": [],
        "started_at": None, "finished_at": None, "result": None, "error": None, "params": {},
    }


job: dict[str, Any] = _fresh_job()


def _job_progress(message: str, pct: float) -> None:
    with job_lock:
        job["message"], job["progress"] = message, round(pct, 3)
        job["log"].append(f"{datetime.now():%H:%M:%S}  {message}")


def _run_retrain(req: RetrainIn, labeled: list[dict[str, Any]]) -> None:
    try:
        version = f"v{db.next_version_number()}"
        result = retrain_mod.retrain(
            model.package(), labeled, version,
            new_row_weight=req.new_row_weight, progress=_job_progress,
        )
        used_ids = result.pop("used_transaction_ids")
        db.insert_version({
            "version": version, "source": "retrain",
            "model_name": result["model_name"], "path": result["path"],
            "threshold": result["threshold"],
            "n_original": result["n_original"], "n_new": result["n_new"],
            "new_row_weight": result["new_row_weight"],
            "metrics_json": result["metrics"],
            "notes": result["recommendation_reason"], "is_active": 0,
        })
        db.mark_used_in_training(used_ids, version)

        activated = False
        if req.auto_activate and result["recommended"]:
            activate_version(version)
            activated = True
            _job_progress(f"{version} activated automatically", 1.0)
        result["activated"] = activated

        with job_lock:
            job.update(state="completed", result=result, finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as exc:
        logger.exception("Retraining failed")
        with job_lock:
            job.update(
                state="failed", error=str(exc) or exc.__class__.__name__,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )


# =====================================================
# APP
# =====================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    bootstrap_versions()
    store.load(HISTORY_PATH)
    logger.info("Database: %s", db.DB_PATH)
    yield


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


app = FastAPI(
    title="FraudShield API",
    description="Transaction fraud scoring with explainable AI, SQLite storage, "
    "Groq analyst briefings and in-app retraining.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
AUTH = [Depends(require_key)]


# ---------------------------------------------------------------- system


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model.pipeline is not None else "model_not_loaded",
        "model": model.model_name,
        "model_version": model.version,
        "threshold": model.threshold,
        "shap_enabled": model.explainer is not None,
        "customers_loaded": len(store.profiles),
        "auth_required": bool(API_KEY),
        "groq_configured": bool(llm.env_key()),
        "database": str(db.DB_PATH),
        "time": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/model/info", tags=["system"], dependencies=AUTH)
def model_info() -> dict[str, Any]:
    return {
        "model_name": model.model_name,
        "model_version": model.version,
        "threshold": model.threshold,
        "features": model.features,
        "test_metrics": (_read_csv_records(REPORT_DIR / "final_test_metrics.csv") or [None])[0],
        "model_comparison": _read_csv_records(REPORT_DIR / "model_comparison.csv"),
        "feature_importance": _read_csv_records(EXPLAIN_DIR / "feature_importance.csv"),
        "loaded_at": model.loaded_at,
    }


# ---------------------------------------------------------------- scoring


@app.post("/predict", response_model=PredictionOut, tags=["scoring"], dependencies=AUTH)
def predict(
    txn: TransactionIn,
    explain: bool = Query(True),
    ai: str = Query("never", pattern="^(never|flagged|always)$",
                    description="Generate a Groq analyst briefing: never, only when flagged, or always"),
    ai_model: Optional[str] = Query(None, description="Groq model id"),
    x_groq_key: Optional[str] = Header(default=None),
) -> PredictionOut:
    scored = score_transaction(txn, explain=explain)
    if txn.persist:
        persist_scored(txn, scored)
    out = scored.out
    if ai == "always" or (ai == "flagged" and out.is_fraud):
        attach_ai(out, txn, x_groq_key, ai_model)
    return out


@app.post("/predict/batch", response_model=BatchOut, tags=["scoring"], dependencies=AUTH)
def predict_batch(batch: BatchIn, explain: bool = Query(False)) -> BatchOut:
    if len(batch.transactions) > MAX_BATCH:
        raise HTTPException(413, f"Batch too large (max {MAX_BATCH})")

    # Process in time order so earlier rows count as history for later rows
    # of the same customer, then return results in the caller's order.
    now = datetime.now()
    order = sorted(
        range(len(batch.transactions)),
        key=lambda i: pd.Timestamp(batch.transactions[i].timestamp or now),
    )
    local_hist: dict[str, pd.DataFrame] = {}
    results: dict[int, PredictionOut] = {}
    errors: list[dict[str, Any]] = []

    for i in order:
        txn = batch.transactions[i]
        try:
            hist = local_hist.get(txn.customer_id)
            if hist is None:
                hist = store.get(txn.customer_id)
            scored = score_transaction(txn, explain=explain, hist_override=hist)
            if batch.persist:
                persist_scored(txn, scored)
            results[i] = scored.out
            local_hist[txn.customer_id] = (
                pd.concat([hist, pd.DataFrame([scored.row])], ignore_index=True)
                if not hist.empty else pd.DataFrame([scored.row])
            )
        except HTTPException as exc:
            errors.append({"row": i, "customer_id": txn.customer_id, "error": exc.detail})
        except Exception as exc:  # keep the rest of the batch going
            errors.append({"row": i, "customer_id": txn.customer_id, "error": str(exc)})

    ordered = [results[i] for i in sorted(results)]
    return BatchOut(
        count=len(ordered),
        flagged=sum(r.is_fraud for r in ordered),
        results=ordered,
        errors=errors,
    )


# ---------------------------------------------------------------- saved transactions


@app.get("/transactions", tags=["transactions"], dependencies=AUTH)
def transactions(
    flagged: Optional[bool] = Query(None, description="true = flagged only, false = not flagged only"),
    review: str = Query("any", pattern="^(any|unreviewed|reviewed|fraud|legit)$"),
    customer: str = "",
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    total, rows = db.list_transactions(flagged, review, customer, limit, offset)
    for r in rows:
        r["predicted_fraud"] = bool(r["predicted_fraud"])
        r["has_ai"] = bool(r["has_ai"])
    return {"total": total, "transactions": rows}


@app.get("/transactions/{transaction_id}", tags=["transactions"], dependencies=AUTH)
def transaction_detail(transaction_id: str) -> dict[str, Any]:
    rec = db.get_transaction(transaction_id)
    if not rec:
        raise HTTPException(404, f"Unknown transaction '{transaction_id}'")
    rec["predicted_fraud"] = bool(rec["predicted_fraud"])
    return rec


@app.post("/transactions/{transaction_id}/review", tags=["transactions"], dependencies=AUTH)
def review_transaction(transaction_id: str, body: ReviewIn) -> dict[str, Any]:
    if not db.set_review(transaction_id, body.label, body.note):
        raise HTTPException(404, f"Unknown transaction '{transaction_id}'")
    return transaction_detail(transaction_id)


@app.post("/transactions/{transaction_id}/ai", tags=["transactions"], dependencies=AUTH)
def transaction_ai(
    transaction_id: str,
    ai_model: Optional[str] = Query(None),
    x_groq_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """(Re)generate the Groq analyst briefing for a saved transaction."""
    rec = db.get_transaction(transaction_id)
    if not rec:
        raise HTTPException(404, f"Unknown transaction '{transaction_id}'")
    result = {
        "fraud_probability": rec["fraud_probability"],
        "threshold": rec["threshold"],
        "is_fraud": bool(rec["predicted_fraud"]),
        "risk_level": rec["risk_level"],
        "recommended_action": rec["recommended_action"],
        "features": rec["features"],
        "reasons": rec["reasons"],
    }
    try:
        text, used = generate_ai(rec, result, x_groq_key, ai_model)
    except llm.GroqError as exc:
        raise HTTPException(exc.status if 400 <= exc.status < 600 else 502, exc.message)
    db.set_ai(transaction_id, text, used)
    return {"transaction_id": transaction_id, "ai_explanation": text, "ai_model": used}


@app.delete("/transactions/{transaction_id}", tags=["transactions"], dependencies=AUTH)
def delete_transaction(transaction_id: str) -> dict[str, Any]:
    customer_id = db.delete_transaction(transaction_id)
    if customer_id is None:
        raise HTTPException(404, f"Unknown transaction '{transaction_id}'")
    store.rebuild(customer_id)  # remove it from the customer's history too
    return {"deleted": transaction_id}


@app.get("/db/stats", tags=["transactions"], dependencies=AUTH)
def db_stats() -> dict[str, Any]:
    return db.stats()


# ---------------------------------------------------------------- AI


@app.get("/ai/status", tags=["ai"], dependencies=AUTH)
def ai_status() -> dict[str, Any]:
    return {"env_key_configured": bool(llm.env_key()), "default_model": llm.env_model()}


@app.get("/ai/models", tags=["ai"], dependencies=AUTH)
def ai_models(x_groq_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    try:
        key = llm.resolve_key(x_groq_key)
        return {"models": llm.list_models(key), "default": llm.env_model()}
    except llm.GroqError as exc:
        raise HTTPException(exc.status if 400 <= exc.status < 600 else 502, exc.message)


# ---------------------------------------------------------------- retraining & versions


@app.get("/retrain/status", tags=["retraining"], dependencies=AUTH)
def retrain_status() -> dict[str, Any]:
    with job_lock:
        snapshot = {**job, "log": list(job["log"])}
    return {
        "data": db.stats(),
        "job": snapshot,
        "active_version": model.version,
        "model_name": model.model_name,
    }


@app.post("/retrain", status_code=202, tags=["retraining"], dependencies=AUTH)
def start_retrain(req: RetrainIn) -> dict[str, Any]:
    labeled = db.labeled_rows()
    if len(labeled) < req.min_new_rows:
        raise HTTPException(
            400,
            f"Only {len(labeled)} reviewed transaction(s); need at least {req.min_new_rows}. "
            "Review saved transactions in the Transactions tab first.",
        )
    with job_lock:
        if job["state"] == "running":
            raise HTTPException(409, "A retraining job is already running.")
        job.clear()
        job.update(_fresh_job())
        job.update(
            state="running", message="Starting",
            started_at=datetime.now().isoformat(timespec="seconds"),
            params=req.model_dump(),
        )
    threading.Thread(target=_run_retrain, args=(req, labeled), daemon=True).start()
    return {"started": True, "reviewed_rows": len(labeled)}


@app.get("/models/versions", tags=["retraining"], dependencies=AUTH)
def model_versions() -> dict[str, Any]:
    return {"active": model.version, "versions": db.list_versions()}


@app.post("/models/versions/{version}/activate", tags=["retraining"], dependencies=AUTH)
def activate(version: str) -> dict[str, Any]:
    with job_lock:
        if job["state"] == "running":
            raise HTTPException(409, "Wait for the retraining job to finish.")
    return activate_version(version)


# ---------------------------------------------------------------- customers


@app.get("/customers", tags=["customers"], dependencies=AUTH)
def list_customers(limit: int = Query(200, ge=1, le=5000), search: str = "") -> dict[str, Any]:
    ids = sorted(store.profiles)
    if search:
        ids = [c for c in ids if search.upper() in c.upper()]
    return {"total": len(ids), "customers": ids[:limit]}


@app.get("/customers/{customer_id}", tags=["customers"], dependencies=AUTH)
def customer_profile(customer_id: str, recent: int = Query(10, ge=0, le=100)) -> dict[str, Any]:
    summary = customer_summary(customer_id, recent)
    if summary is None:
        raise HTTPException(404, f"Unknown customer '{customer_id}'")
    return summary


# ---------------------------------------------------------------- analytics


@app.get("/analytics/summary", tags=["analytics"], dependencies=AUTH)
def analytics_summary() -> dict[str, Any]:
    df = store.raw
    out: dict[str, Any] = {"scored_by_api": len(recent_predictions), "database": db.stats()}
    if df.empty or "Class" not in df.columns:
        return out

    fraud = df[df["Class"] == 1]
    daily = (
        df.assign(day=df["timestamp"].dt.date.astype(str))
        .groupby("day")["Class"].agg(transactions="count", fraud="sum")
        .reset_index()
    )
    daily["fraud_rate"] = (daily["fraud"] / daily["transactions"] * 100).round(2)

    def breakdown(col: str) -> list[dict[str, Any]]:
        g = df.groupby(col)["Class"].agg(transactions="count", fraud="sum").reset_index()
        g["fraud_rate"] = (g["fraud"] / g["transactions"] * 100).round(2)
        return g.sort_values("fraud", ascending=False).to_dict(orient="records")

    fraud_amount = fraud.loc[fraud["currency"] == "INR", "amount"]
    out.update(
        {
            "total_transactions": int(len(df)),
            "fraud_transactions": int(len(fraud)),
            "fraud_rate_pct": round(float(df["Class"].mean() * 100), 2),
            "customers": int(df["customer_id"].nunique()),
            "inr_fraud_amount": round(float(fraud_amount.sum()), 2),
            "date_from": df["timestamp"].min().isoformat(sep=" "),
            "date_to": df["timestamp"].max().isoformat(sep=" "),
            "by_fraud_type": (
                fraud.groupby("fraud_type").size().rename("cases").reset_index()
                .to_dict(orient="records")
                if "fraud_type" in fraud.columns else []
            ),
            "by_transaction_type": breakdown("transaction_type"),
            "by_currency": breakdown("currency"),
            "daily": daily.to_dict(orient="records"),
        }
    )
    return out


@app.get("/predictions/recent", tags=["analytics"], dependencies=AUTH)
def recent(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    items = list(recent_predictions)[-limit:][::-1]
    return {"count": len(items), "predictions": items}