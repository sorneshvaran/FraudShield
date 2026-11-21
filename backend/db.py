"""
FraudShield - SQLite persistence
================================

One file database (default: data/fraudshield.db, override with the
FRAUDSHIELD_DB environment variable) holding:

* transactions    every transaction you chose to save, with the exact
                  engineered features the model saw, its prediction, the
                  AI explanation, and your review label (the ground truth
                  used for retraining)
* customers       profiles of customers that were not in the original CSV
* model_versions  registry of trained models (active / candidate)

Only the python standard library is used.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("FRAUDSHIELD_DB", BASE_DIR / "data" / "fraudshield.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id             TEXT PRIMARY KEY,
    registered_latitude     REAL,
    registered_longitude    REAL,
    account_creation_date   TEXT,
    created_at              TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id          TEXT UNIQUE NOT NULL,
    created_at              TEXT NOT NULL,
    timestamp               TEXT NOT NULL,
    customer_id             TEXT NOT NULL,
    amount                  REAL NOT NULL,
    currency                TEXT NOT NULL,
    merchant_id             TEXT,
    payee_id                TEXT,
    device_id               TEXT,
    merchant_category       TEXT,
    transaction_type        TEXT,
    transaction_subtype     TEXT,
    transaction_status      TEXT NOT NULL DEFAULT 'SUCCESS',
    card_present            INTEGER,
    is_international        INTEGER,
    transaction_latitude    REAL,
    transaction_longitude   REAL,

    features_json           TEXT NOT NULL,
    reasons_json            TEXT,

    fraud_probability       REAL NOT NULL,
    threshold               REAL NOT NULL,
    predicted_fraud         INTEGER NOT NULL,
    risk_level              TEXT NOT NULL,
    recommended_action      TEXT,
    model_name              TEXT,
    model_version           TEXT,

    ai_explanation          TEXT,
    ai_model                TEXT,
    ai_generated_at         TEXT,

    review_label            INTEGER,          -- 1 = confirmed fraud, 0 = confirmed legit, NULL = not reviewed
    review_note             TEXT,
    reviewed_at             TEXT,
    used_in_training        TEXT              -- model version that last trained on this row
);

CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_review   ON transactions(review_label);
CREATE INDEX IF NOT EXISTS idx_txn_time     ON transactions(timestamp);

CREATE TABLE IF NOT EXISTS model_versions (
    version         TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    source          TEXT,
    model_name      TEXT,
    path            TEXT NOT NULL,
    threshold       REAL,
    n_original      INTEGER,
    n_new           INTEGER,
    new_row_weight  REAL,
    metrics_json    TEXT,
    notes           TEXT,
    is_active       INTEGER NOT NULL DEFAULT 0
);
"""

TXN_COLUMNS = [
    "transaction_id", "created_at", "timestamp", "customer_id", "amount", "currency",
    "merchant_id", "payee_id", "device_id", "merchant_category", "transaction_type",
    "transaction_subtype", "transaction_status", "card_present", "is_international",
    "transaction_latitude", "transaction_longitude", "features_json", "reasons_json",
    "fraud_probability", "threshold", "predicted_fraud", "risk_level",
    "recommended_action", "model_name", "model_version",
]

# Columns returned by list queries (no big JSON blobs).
LIST_COLUMNS = (
    "transaction_id, created_at, timestamp, customer_id, amount, currency, "
    "transaction_type, merchant_category, payee_id, merchant_id, device_id, "
    "fraud_probability, threshold, predicted_fraud, risk_level, model_version, "
    "review_label, reviewed_at, used_in_training, "
    "(ai_explanation IS NOT NULL) AS has_ai"
)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def _row(r: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(r) if r is not None else None


# ----------------------------------------------------------------- customers


def upsert_customer(profile: dict[str, Any]) -> None:
    created = profile.get("account_creation_date")
    if isinstance(created, datetime):
        created = created.strftime("%Y-%m-%d %H:%M:%S")
    with connect() as c:
        c.execute(
            """INSERT OR IGNORE INTO customers
               (customer_id, registered_latitude, registered_longitude, account_creation_date, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                profile["customer_id"], profile["registered_latitude"],
                profile["registered_longitude"], created, now_str(),
            ),
        )


def list_customers() -> list[dict[str, Any]]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM customers")]


# -------------------------------------------------------------- transactions


def insert_transaction(rec: dict[str, Any]) -> None:
    """rec must contain the TXN_COLUMNS keys (missing ones become NULL);
    features/reasons may be passed as python objects."""
    rec = dict(rec)
    rec.setdefault("created_at", now_str())
    for key in ("features_json", "reasons_json"):
        if key in rec and not isinstance(rec[key], str):
            rec[key] = json.dumps(rec[key], default=str)
    values = [rec.get(col) for col in TXN_COLUMNS]
    placeholders = ",".join("?" for _ in TXN_COLUMNS)
    with connect() as c:
        c.execute(
            f"INSERT INTO transactions ({','.join(TXN_COLUMNS)}) VALUES ({placeholders})",
            values,
        )


def get_transaction(transaction_id: str) -> Optional[dict[str, Any]]:
    with connect() as c:
        r = c.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
    out = _row(r)
    if out:
        out["features"] = json.loads(out.pop("features_json") or "{}")
        out["reasons"] = json.loads(out.pop("reasons_json") or "[]")
    return out


def list_transactions(
    flagged: Optional[bool] = None,
    review: str = "any",           # any | unreviewed | reviewed | fraud | legit
    customer: str = "",
    limit: int = 100,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    where, args = [], []
    if flagged is not None:
        where.append("predicted_fraud = ?")
        args.append(int(flagged))
    if review == "unreviewed":
        where.append("review_label IS NULL")
    elif review == "reviewed":
        where.append("review_label IS NOT NULL")
    elif review == "fraud":
        where.append("review_label = 1")
    elif review == "legit":
        where.append("review_label = 0")
    if customer:
        where.append("customer_id LIKE ?")
        args.append(f"%{customer}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    with connect() as c:
        total = c.execute(f"SELECT COUNT(*) FROM transactions {clause}", args).fetchone()[0]
        rows = c.execute(
            f"SELECT {LIST_COLUMNS} FROM transactions {clause} "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            [*args, limit, offset],
        ).fetchall()
    return total, [dict(r) for r in rows]


def set_review(transaction_id: str, label: Optional[int], note: Optional[str]) -> bool:
    with connect() as c:
        cur = c.execute(
            """UPDATE transactions
               SET review_label = ?, review_note = ?, reviewed_at = ?
               WHERE transaction_id = ?""",
            (label, note, now_str() if label is not None else None, transaction_id),
        )
        return cur.rowcount > 0


def set_ai(transaction_id: str, text: str, model: str) -> bool:
    with connect() as c:
        cur = c.execute(
            """UPDATE transactions
               SET ai_explanation = ?, ai_model = ?, ai_generated_at = ?
               WHERE transaction_id = ?""",
            (text, model, now_str(), transaction_id),
        )
        return cur.rowcount > 0


def delete_transaction(transaction_id: str) -> Optional[str]:
    """Returns the customer_id of the deleted row (or None)."""
    with connect() as c:
        r = c.execute(
            "SELECT customer_id FROM transactions WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
        if r is None:
            return None
        c.execute("DELETE FROM transactions WHERE transaction_id = ?", (transaction_id,))
        return r["customer_id"]


def history_rows(customer_id: Optional[str] = None) -> list[dict[str, Any]]:
    sql = (
        "SELECT customer_id, timestamp, amount, currency, transaction_status, "
        "device_id, merchant_id, payee_id FROM transactions"
    )
    args: list[Any] = []
    if customer_id:
        sql += " WHERE customer_id = ?"
        args.append(customer_id)
    sql += " ORDER BY timestamp"
    with connect() as c:
        return [dict(r) for r in c.execute(sql, args)]


def stats() -> dict[str, Any]:
    with connect() as c:
        r = c.execute(
            """SELECT
                 COUNT(*)                                             AS total,
                 COALESCE(SUM(predicted_fraud), 0)                    AS flagged,
                 COALESCE(SUM(review_label IS NOT NULL), 0)           AS reviewed,
                 COALESCE(SUM(review_label = 1), 0)                   AS confirmed_fraud,
                 COALESCE(SUM(review_label = 0), 0)                   AS confirmed_legit,
                 COALESCE(SUM(predicted_fraud = 1 AND review_label IS NULL), 0) AS flagged_unreviewed,
                 COALESCE(SUM(review_label IS NOT NULL AND used_in_training IS NULL), 0) AS labeled_new,
                 COALESCE(SUM(predicted_fraud = 1 AND review_label = 0), 0) AS false_positives,
                 COALESCE(SUM(predicted_fraud = 0 AND review_label = 1), 0) AS missed_fraud,
                 COALESCE(SUM(ai_explanation IS NOT NULL), 0)         AS with_ai
               FROM transactions"""
        ).fetchone()
    return dict(r)


# ---------------------------------------------------------------- training


def labeled_rows() -> list[dict[str, Any]]:
    """Every reviewed transaction with its stored features and label."""
    with connect() as c:
        rows = c.execute(
            """SELECT transaction_id, timestamp, customer_id, features_json,
                      review_label AS label
               FROM transactions WHERE review_label IS NOT NULL
               ORDER BY timestamp"""
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d.pop("features_json") or "{}")
        out.append(d)
    return out


def mark_used_in_training(transaction_ids: list[str], version: str) -> None:
    if not transaction_ids:
        return
    with connect() as c:
        c.executemany(
            "UPDATE transactions SET used_in_training = ? WHERE transaction_id = ?",
            [(version, t) for t in transaction_ids],
        )


# ----------------------------------------------------------- model versions


def insert_version(rec: dict[str, Any]) -> None:
    rec = dict(rec)
    if not isinstance(rec.get("metrics_json"), str):
        rec["metrics_json"] = json.dumps(rec.get("metrics_json") or {}, default=str)
    cols = [
        "version", "created_at", "source", "model_name", "path", "threshold",
        "n_original", "n_new", "new_row_weight", "metrics_json", "notes", "is_active",
    ]
    rec.setdefault("created_at", now_str())
    rec.setdefault("is_active", 0)
    with connect() as c:
        c.execute(
            f"INSERT OR REPLACE INTO model_versions ({','.join(cols)}) "
            f"VALUES ({','.join('?' for _ in cols)})",
            [rec.get(k) for k in cols],
        )


def list_versions() -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM model_versions ORDER BY created_at DESC, version DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
        d["is_active"] = bool(d["is_active"])
        out.append(d)
    return out


def get_version(version: str) -> Optional[dict[str, Any]]:
    for v in list_versions():
        if v["version"] == version:
            return v
    return None


def set_active_version(version: str) -> None:
    with connect() as c:
        c.execute("UPDATE model_versions SET is_active = 0")
        c.execute("UPDATE model_versions SET is_active = 1 WHERE version = ?", (version,))


def next_version_number() -> int:
    with connect() as c:
        rows = c.execute("SELECT version FROM model_versions").fetchall()
    nums = [int(r["version"][1:]) for r in rows if r["version"][1:].isdigit()]
    return (max(nums) + 1) if nums else 1