"""
FraudShield - Streamlit dashboard
=================================

1) Start the API   (from the FraudShield/ folder)
       uvicorn backend.api:app --port 8000

2) Start this app  (from the FraudShield/ folder)
       streamlit run frontend/app.py

Settings live in the .env file in FraudShield/ (nothing to type in the app):
    GROQ_API_KEY          your Groq key, read by the API for AI recommendations
    GROQ_MODEL            optional default Groq model
    FRAUDSHIELD_API_URL   only if the API is NOT at http://127.0.0.1:8000
    FRAUDSHIELD_API_KEY   only if you started the API with a key
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_URL = os.getenv("FRAUDSHIELD_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("FRAUDSHIELD_API_KEY", "")

TEST_DATA = BASE_DIR / "data" / "fraud_intelligence" / "test_data.csv"
EXPLAIN_DIR = BASE_DIR / "data" / "explainability"
REPORT_DIR = BASE_DIR / "data" / "fraud_intelligence" / "model_reports"
INTEL_DIR = BASE_DIR / "data" / "fraud_intelligence"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

RISK_STYLE = {
    "LOW": ("success", "🟢"),
    "MEDIUM": ("warning", "🟡"),
    "HIGH": ("error", "🟠"),
    "CRITICAL": ("error", "🔴"),
}
RED, GREEN = "#EF4444", "#22C55E"

LOCATIONS = {  # name -> (lat, lon)
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Dubai": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198),
    "London": (51.5072, -0.1276),
    "New York": (40.7128, -74.0060),
}
TXN_TYPES = ["UPI", "Debit Card", "Credit Card", "NetBanking", "Wallet"]
MERCHANT_CATEGORIES = [
    "Groceries", "Electronics", "Fashion", "Travel", "Food & Dining", "Fuel",
    "Utilities", "Entertainment", "Healthcare", "Jewellery", "Money Transfer", "Other",
]
CURRENCIES = ["INR", "USD", "EUR", "GBP", "AED", "SGD"]

SERVER_DEFAULT_MODEL = "(server default)"
CUSTOM_MODEL = "Custom…"

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container, div[data-testid="stMainBlockContainer"] {padding-top: 4.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
          background: rgba(148,163,184,.08);
          border: 1px solid rgba(148,163,184,.18);
          padding: .7rem .9rem; border-radius: .7rem;
      }
      .fs-title {font-size: 2rem; font-weight: 700; line-height: 1.3; margin-bottom: 0;}
      .fs-sub {color: #94A3B8; margin-top: -.3rem;}
      .reason {padding: .45rem .75rem; margin: .25rem 0; border-radius: .5rem;
               border-left: 4px solid; background: rgba(148,163,184,.08);}
      .reason.up {border-color: #EF4444;}
      .reason.down {border-color: #22C55E;}
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state


# =====================================================
# API CLIENT
# =====================================================


class ApiError(Exception):
    pass


def call_api(
    method: str, path: str, url: str, key: str,
    json: Optional[dict] = None, params: Optional[dict] = None,
    timeout: int = 60, extra_headers: Optional[dict] = None,
) -> Any:
    headers = {"X-API-Key": key} if key else {}
    headers.update(extra_headers or {})
    try:
        r = requests.request(
            method, url.rstrip("/") + path, json=json, params=params,
            headers=headers, timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {url}. Is it running?")
    except requests.exceptions.Timeout:
        raise ApiError("The API took too long to respond.")

    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(
                f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail
            )
        raise ApiError(str(detail))
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def cached_get(path: str, url: str, key: str, params: Optional[tuple] = None) -> Any:
    return call_api("GET", path, url, key, params=dict(params) if params else None)


def groq_model() -> str:
    pick = ss.get("groq_model_pick", SERVER_DEFAULT_MODEL)
    if pick == CUSTOM_MODEL:
        return (ss.get("groq_model_custom") or "").strip()
    return "" if pick == SERVER_DEFAULT_MODEL else pick


def api(method: str, path: str, json: Optional[dict] = None, params: Optional[dict] = None,
        timeout: int = 60) -> Any:
    """Call the FraudShield API."""
    return call_api(method, path, API_URL, API_KEY, json=json, params=params, timeout=timeout)


def get(path: str, **params: Any) -> Any:
    return cached_get(path, API_URL, API_KEY, tuple(sorted(params.items())) or None)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("### 🛡️ FraudShield")
    if st.button("↻ Refresh data", width="stretch"):
        st.cache_data.clear()

try:
    health = call_api("GET", "/health", API_URL, API_KEY, timeout=5)
except ApiError as exc:
    st.error(f"**{exc}**")
    st.code(
        "# from the FraudShield/ folder\n"
        "uvicorn backend.api:app --port 8000",
        language="bash",
    )
    st.stop()

try:
    ai_status = get("/ai/status")
except ApiError:
    ai_status = {"env_key_configured": False, "default_model": ""}

AI_READY = bool(ai_status["env_key_configured"])

with st.sidebar:
    st.success("API connected")
    st.caption(f"**Model:** {health['model']}  ·  `{health.get('model_version')}`")
    st.caption(f"**Threshold:** {health['threshold']:.3f}  ·  **Customers:** {health['customers_loaded']:,}")
    st.caption(f"**Database:** `{Path(health['database']).name}`")

    st.divider()
    st.markdown("##### 🤖 AI recommendations (Groq)")
    if AI_READY:
        models = ss.get("groq_models") or []
        ss.setdefault("groq_model_pick", SERVER_DEFAULT_MODEL)
        options = [SERVER_DEFAULT_MODEL, *models, CUSTOM_MODEL]
        if ss.groq_model_pick not in options:
            ss.groq_model_pick = SERVER_DEFAULT_MODEL
        st.selectbox("Model", options, key="groq_model_pick")
        if ss.groq_model_pick == CUSTOM_MODEL:
            st.text_input("Groq model id", key="groq_model_custom", placeholder="llama-3.3-70b-versatile")
        elif ss.groq_model_pick == SERVER_DEFAULT_MODEL and ai_status.get("default_model"):
            st.caption(f"Default: `{ai_status['default_model']}`")
        if st.button("Load models from Groq", width="stretch"):
            try:
                ss.groq_models = api("GET", "/ai/models")["models"]
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
        st.checkbox("Auto-write a recommendation for flagged transactions", value=True, key="auto_ai")
    else:
        st.warning(
            "AI recommendations are off. Add `GROQ_API_KEY=...` to the `.env` file in the "
            "FraudShield folder and restart the API."
        )

    st.divider()
    st.caption(
        "Risk levels: 🟢 LOW · 🟡 MEDIUM (step-up auth) · 🟠 HIGH (analyst review) · 🔴 CRITICAL (block)"
    )

st.markdown('<p class="fs-title">🛡️ FraudShield</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="fs-sub">Enter a transaction · get a fraud verdict and an AI explanation · '
    'review it · retrain the model on your own data</p>',
    unsafe_allow_html=True,
)

(tab_check, tab_batch, tab_txns, tab_retrain,
 tab_dash, tab_model, tab_customer) = st.tabs([
    "🔍 Check Transaction", "📦 Batch Scoring", "🗂️ Transactions", "🔁 Retrain",
    "📊 Dashboard", "🧠 Model Insights", "👤 Customer 360",
])


# =====================================================
# SHARED HELPERS / CALLBACKS
# =====================================================


def money(v: Optional[float], cur: str = "") -> str:
    return "—" if v is None else f"{v:,.2f} {cur}".strip()


def review_text(label: Any) -> str:
    if label is None or pd.isna(label):
        return "—"
    return "🚨 Fraud" if int(label) == 1 else "✅ Legit"


def shap_chart(reasons: list[dict], unit: Optional[str]) -> alt.Chart:
    df = pd.DataFrame(reasons)
    df["impact"] = df["shap_value"]
    df["effect"] = df["impact"].apply(lambda x: "Raises risk" if x > 0 else "Lowers risk")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("impact:Q", title=f"Impact on fraud score ({unit or 'SHAP'})"),
            y=alt.Y("label:N", sort=alt.EncodingSortField("impact", order="descending"), title=None),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, GREEN]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["label", "text", alt.Tooltip("impact:Q", format="+.3f")],
        )
        .properties(height=max(120, 34 * len(df)))
    )


def flash(scope: str, kind: str, message: str) -> None:
    ss.flash = {"scope": scope, "kind": kind, "message": message}


def show_flash(scope: str) -> None:
    f = ss.get("flash")
    if f and f["scope"] == scope:
        getattr(st, f["kind"])(f["message"])
        ss.flash = None


def cb_review(tid: str, label: Optional[int], scope: str) -> None:
    note = (ss.get(f"{scope}_note_{tid}") or "").strip() or None
    try:
        api("POST", f"/transactions/{tid}/review", json={"label": label, "note": note})
        st.cache_data.clear()
        flash(scope, "success", {1: "Marked as FRAUD.", 0: "Marked as LEGIT.", None: "Review cleared."}[label])
    except ApiError as exc:
        flash(scope, "error", str(exc))


def cb_ai(tid: str, scope: str) -> None:
    try:
        params = {"ai_model": groq_model()} if groq_model() else None
        api("POST", f"/transactions/{tid}/ai", params=params, timeout=90)
        flash(scope, "success", "AI recommendation updated.")
    except ApiError as exc:
        flash(scope, "error", str(exc))


def render_assessment(d: dict) -> None:
    """Banner + metrics. d needs: is_fraud, risk_level, fraud_probability, threshold,
    recommended_action, transaction_id."""
    level = d["risk_level"]
    kind, icon = RISK_STYLE[level]
    prob = d["fraud_probability"]
    getattr(st, kind)(
        f"{icon} **{level} RISK** — "
        f"{'FRAUD ALERT' if d['is_fraud'] else 'Looks legitimate'}  ·  {d['recommended_action']}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud probability", f"{prob:.2%}")
    c2.metric("Decision threshold", f"{d['threshold']:.2%}")
    c3.metric("Verdict", "FRAUD" if d["is_fraud"] else "LEGIT")
    c4.metric("Transaction ID", d["transaction_id"][-8:])
    st.progress(min(max(prob, 0.0), 1.0))


def render_reasons(reasons: list[dict], unit: Optional[str], features: dict) -> None:
    if reasons:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Why this score?")
            for r in reasons:
                up = r["direction"] == "increases_risk"
                st.markdown(
                    f'<div class="reason {"up" if up else "down"}">'
                    f'{"▲" if up else "▼"} <b>{r["label"]}</b> — {r["text"]}</div>',
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown("#### Feature impact")
            st.altair_chart(shap_chart(reasons, unit), width="stretch")
    else:
        st.info("No explanation available (SHAP disabled).")

    with st.expander("Engineered features sent to the model"):
        st.dataframe(
            pd.DataFrame([{"feature": k, "value": v} for k, v in features.items()]),
            hide_index=True, width="stretch",
        )


def render_ai(tid: Optional[str], text: Optional[str], model_used: Optional[str],
              error: Optional[str], is_flagged: bool, scope: str) -> None:
    st.markdown("#### 🤖 AI recommendation")
    show_flash(scope)
    if text:
        st.markdown(text)
        st.caption(f"Written by Groq · `{model_used}` · an assistant's summary of the model's output, not a final decision.")
    elif error:
        st.warning(error)
    elif is_flagged and not AI_READY:
        st.info("Add `GROQ_API_KEY` to your `.env` file and restart the API to get an AI explanation of why this was flagged.")
    elif tid is None:
        st.caption("Save the transaction to the database to generate an AI recommendation on demand.")

    if tid is not None and AI_READY:
        st.button(
            "🔄 Regenerate" if text else "✨ Generate AI recommendation",
            on_click=cb_ai, args=(tid, scope), key=f"{scope}_ai_{tid}",
        )


def render_review(tid: str, label: Optional[int], note: Optional[str],
                  is_flagged: bool, scope: str) -> None:
    st.markdown("#### 🧑‍⚖️ Your verdict")
    st.caption(
        "Confirm what really happened. **Only transactions you review are used when you "
        "retrain the model** — the model's own guesses are never treated as the truth."
    )
    if label == 1:
        st.error("Confirmed **FRAUD**")
    elif label == 0:
        st.success("Confirmed **LEGIT**")
    else:
        st.info("Not reviewed yet")
    if label is not None and bool(label) != bool(is_flagged):
        st.warning(
            "The model disagreed with your verdict here — corrections like this are the most "
            "valuable examples for retraining."
        )

    st.text_input("Note (optional)", value=note or "", key=f"{scope}_note_{tid}", max_chars=500)
    c1, c2, c3 = st.columns(3)
    c1.button("🚨 It's fraud", on_click=cb_review, args=(tid, 1, scope), key=f"{scope}_f_{tid}", width="stretch")
    c2.button("✅ It's legit", on_click=cb_review, args=(tid, 0, scope), key=f"{scope}_l_{tid}", width="stretch")
    c3.button("↩ Clear", on_click=cb_review, args=(tid, None, scope), key=f"{scope}_c_{tid}", width="stretch")


# =====================================================
# TAB 1 - CHECK A TRANSACTION
# =====================================================

PRESETS = {
    "🛒 Everyday purchase": dict(
        amount=450.0, payee="None", device="Same as last device", loc="Customer's home",
        hour=dtime(12, 30), card_present=True, intl=False, ttype="Debit Card", cat="Groceries",
        merchant="Known merchant",
    ),
    "🔓 Account takeover": dict(
        amount=48000.0, payee="New payee", device="New device", loc="London",
        hour=dtime(3, 15), card_present=False, intl=True, ttype="NetBanking", cat="Money Transfer",
        merchant="None",
    ),
    "🐴 Money mule transfer": dict(
        amount=95000.0, payee="New payee", device="Same as last device", loc="Customer's home",
        hour=dtime(2, 40), card_present=False, intl=False, ttype="UPI", cat="Money Transfer",
        merchant="None",
    ),
    "💳 Card-not-present abroad": dict(
        amount=18000.0, payee="None", device="New device", loc="Singapore",
        hour=dtime(23, 50), card_present=False, intl=True, ttype="Credit Card", cat="Electronics",
        merchant="New merchant",
    ),
}


def apply_preset(name: str) -> None:
    p = PRESETS[name]
    ss.chk_amount, ss.chk_payee, ss.chk_device = p["amount"], p["payee"], p["device"]
    ss.chk_loc, ss.chk_time = p["loc"], p["hour"]
    ss.chk_card_present, ss.chk_intl = p["card_present"], p["intl"]
    ss.chk_ttype, ss.chk_cat, ss.chk_merchant = p["ttype"], p["cat"], p["merchant"]


for key, default in dict(
    chk_amount=1500.0, chk_payee="None", chk_device="Same as last device",
    chk_loc="Customer's home", chk_time=dtime(14, 0), chk_card_present=True,
    chk_intl=False, chk_ttype="UPI", chk_cat="Groceries", chk_merchant="Known merchant",
    chk_status="SUCCESS", chk_persist=True,
).items():
    ss.setdefault(key, default)

with tab_check:
    try:
        customers = get("/customers", limit=5000)["customers"]
    except ApiError as exc:
        st.error(str(exc))
        customers = []

    if not customers:
        st.warning("No customers found in the history file (data/processed_data.csv).")
    else:
        top = st.columns([2, 5])
        customer_id = top[0].selectbox("Customer", customers, key="chk_customer")
        try:
            prof = get(f"/customers/{customer_id}", recent=0)
        except ApiError as exc:
            st.error(str(exc))
            prof = None

        if prof:
            with top[1]:
                m = st.columns(5)
                m[0].metric("Transactions", prof["transaction_count"])
                m[1].metric("Avg amount", money(prof["avg_amount"], prof["primary_currency"]))
                m[2].metric("Known payees", prof["known_payees"])
                m[3].metric("Last device", prof["last_device_id"] or "—")
                m[4].metric(
                    "Home",
                    f"{prof['registered_latitude']:.2f}, {prof['registered_longitude']:.2f}",
                )

            st.markdown("**Quick scenarios**")
            pc = st.columns(len(PRESETS))
            for col, name in zip(pc, PRESETS):
                col.button(name, on_click=apply_preset, args=(name,), width="stretch")

            st.divider()
            a, b, c = st.columns(3)

            with a:
                st.markdown("##### Payment")
                amount = st.number_input("Amount", min_value=1.0, step=100.0, key="chk_amount")
                currency = st.selectbox(
                    "Currency", CURRENCIES,
                    index=CURRENCIES.index(prof["primary_currency"]) if prof["primary_currency"] in CURRENCIES else 0,
                    key=f"chk_currency_{customer_id}",
                )
                ttype = st.selectbox("Type", TXN_TYPES, key="chk_ttype")
                cat = st.selectbox("Merchant category", MERCHANT_CATEGORIES, key="chk_cat")
                status = st.selectbox("Status", ["SUCCESS", "FAILED"], key="chk_status")

            with b:
                st.markdown("##### Counterparty & device")
                merchant_mode = st.selectbox(
                    "Merchant", ["None", "Known merchant", "New merchant"], key="chk_merchant"
                )
                payee_mode = st.selectbox(
                    "Payee", ["None", "Known payee", "New payee"], key="chk_payee"
                )
                device_mode = st.selectbox(
                    "Device", ["Same as last device", "New device"], key="chk_device"
                )
                card_present = st.checkbox("Card physically present", key="chk_card_present")
                intl = st.checkbox("International transaction", key="chk_intl")

            with c:
                st.markdown("##### When & where")
                d = st.date_input("Date", value=datetime.now().date(), key="chk_date")
                t = st.time_input("Time", key="chk_time")
                loc_choice = st.selectbox(
                    "Location", ["Customer's home", *LOCATIONS, "Custom"], key="chk_loc"
                )
                if loc_choice == "Custom":
                    lat = st.number_input("Latitude", -90.0, 90.0, 20.0, format="%.4f")
                    lon = st.number_input("Longitude", -180.0, 180.0, 78.0, format="%.4f")
                elif loc_choice == "Customer's home":
                    lat, lon = prof["registered_latitude"], prof["registered_longitude"]
                else:
                    lat, lon = LOCATIONS[loc_choice]
                st.checkbox(
                    "Save to database",
                    key="chk_persist",
                    help="Saved transactions can be reviewed, used for retraining, and count as "
                         "history for this customer's later checks (velocity, new payee, baseline).",
                )

            if st.button("🔎 Analyze transaction", type="primary", width="stretch"):
                rnd = random.Random()

                def pick(known: list[str], make_new: bool, prefix: str) -> Optional[str]:
                    if make_new:
                        return f"{prefix}{rnd.randint(900000, 999999)}"
                    return known[0] if known else None

                merchant_id = (
                    None if merchant_mode == "None"
                    else pick(prof["known_merchant_ids"], merchant_mode == "New merchant", "MER_NEW_")
                )
                payee_id = (
                    None if payee_mode == "None"
                    else pick(prof["known_payee_ids"], payee_mode == "New payee", "PAY_NEW_")
                )
                device_id = (
                    f"DEV_NEW_{rnd.randint(900000, 999999)}" if device_mode == "New device"
                    else prof["last_device_id"]
                )

                payload = dict(
                    customer_id=customer_id,
                    amount=float(amount),
                    currency=currency,
                    timestamp=datetime.combine(d, t).isoformat(),
                    merchant_id=merchant_id,
                    payee_id=payee_id,
                    device_id=device_id,
                    merchant_category=cat,
                    transaction_type=ttype,
                    transaction_status=status,
                    card_present=int(card_present),
                    is_international=int(intl),
                    transaction_latitude=float(lat),
                    transaction_longitude=float(lon),
                    persist=bool(ss.chk_persist),
                )
                params = {"ai": "flagged" if (AI_READY and ss.get("auto_ai", True)) else "never"}
                if groq_model():
                    params["ai_model"] = groq_model()
                with st.spinner("Scoring…"):
                    try:
                        ss.last_result = api("POST", "/predict", json=payload, params=params, timeout=90)
                        ss.last_error = None
                        st.cache_data.clear()
                    except ApiError as exc:
                        ss.last_result = None
                        ss.last_error = str(exc)

            if ss.get("last_error"):
                st.error(ss.last_error)
            elif ss.get("last_result"):
                res = ss.last_result
                st.divider()
                render_assessment(res)
                render_reasons(res["reasons"], res.get("explanation_unit"), res["features"])
                st.divider()

                tid = res["transaction_id"] if res["saved"] else None
                text, model_used, err = res.get("ai_explanation"), res.get("ai_model"), res.get("ai_error")
                review_label, review_note = None, None
                if tid:
                    try:  # the database is the source of truth after "Generate / Review" clicks
                        detail = api("GET", f"/transactions/{tid}")
                        text = detail.get("ai_explanation") or text
                        model_used = detail.get("ai_model") or model_used
                        err = None if text else err
                        review_label, review_note = detail.get("review_label"), detail.get("review_note")
                    except ApiError:
                        pass

                left, right = st.columns(2)
                with left:
                    render_ai(tid, text, model_used, err, res["is_fraud"], "chk")
                with right:
                    if tid:
                        render_review(tid, review_label, review_note, res["is_fraud"], "chk")
                    else:
                        st.markdown("#### 🧑‍⚖️ Your verdict")
                        st.info("This transaction was not saved. Tick **Save to database** to review it "
                                "and use it for retraining.")


# =====================================================
# TAB 2 - BATCH SCORING
# =====================================================

BATCH_REQUIRED = [
    "customer_id", "amount", "currency", "transaction_latitude", "transaction_longitude",
]
BATCH_OPTIONAL = [
    "timestamp", "merchant_id", "payee_id", "device_id", "merchant_category",
    "transaction_type", "transaction_subtype", "transaction_status",
    "card_present", "is_international", "label",
]
BATCH_FIELDS = BATCH_REQUIRED + BATCH_OPTIONAL


def frame_to_transactions(df: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in df[[c for c in BATCH_FIELDS if c in df.columns]].to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if pd.isna(v) or (isinstance(v, str) and not v.strip()):
                continue
            if k == "timestamp":
                v = pd.Timestamp(v).isoformat()
            elif k in ("amount", "transaction_latitude", "transaction_longitude"):
                v = float(v)
            elif k == "label":
                clean["review_label"] = int(v)
                continue
            elif k in ("card_present", "is_international"):
                v = str(v)
            else:
                v = str(v) if not isinstance(v, str) else v
            clean[k] = v
        rows.append(clean)
    return rows


def run_batch(df: pd.DataFrame, persist: bool) -> Optional[pd.DataFrame]:
    txns = frame_to_transactions(df)
    if persist:  # per-row persist flag is irrelevant in batch; the batch flag decides
        pass
    results, errors = [], []
    bar = st.progress(0.0, text="Scoring…")
    chunk = 400
    for start in range(0, len(txns), chunk):
        part = txns[start:start + chunk]
        try:
            out = api("POST", "/predict/batch", json={"transactions": part, "persist": persist}, timeout=300)
        except ApiError as exc:
            st.error(str(exc))
            return None
        # results come back in order of successfully scored rows
        failed_rows = {e["row"] for e in out["errors"]}
        ok_idx = [i for i in range(len(part)) if i not in failed_rows]
        for i, r in zip(ok_idx, out["results"]):
            results.append((start + i, r))
        errors += [{**e, "row": e["row"] + start} for e in out["errors"]]
        bar.progress(min((start + chunk) / len(txns), 1.0),
                     text=f"Scored {min(start + chunk, len(txns))}/{len(txns)}")
    bar.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} rows could not be scored"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
    if not results:
        st.warning("No rows were scored.")
        return None

    idx = [i for i, _ in results]
    res = df.iloc[idx].copy()
    res["transaction_id_api"] = [r["transaction_id"] for _, r in results]
    res["fraud_probability"] = [r["fraud_probability"] for _, r in results]
    res["predicted_fraud"] = [r["is_fraud"] for _, r in results]
    res["risk_level"] = [r["risk_level"] for _, r in results]
    if persist:
        st.cache_data.clear()
    return res


with tab_batch:
    st.markdown("Score many transactions at once from a CSV or a sample of your held-out test set.")
    source = st.radio("Source", ["Upload CSV", "Sample from test set"], horizontal=True)

    if source == "Sample from test set":
        if not TEST_DATA.exists():
            st.warning(f"Test set not found at {TEST_DATA}")
        else:
            c1, c2 = st.columns(2)
            n = c1.slider("Rows", 20, 1000, 200, step=20)
            mix = c2.slider("Share of known fraud cases", 0, 100, 20,
                            help="Enrich the sample with real fraud rows so results are interesting.")
            st.caption("Sample rows are already part of the original dataset, so they are scored "
                       "for evaluation only and are never saved or used for retraining.")
            if st.button("Load sample", key="load_sample"):
                full = pd.read_csv(TEST_DATA, parse_dates=["timestamp"])
                n_fraud = min(int(n * mix / 100), int((full["Class"] == 1).sum()))
                part = pd.concat([
                    full[full["Class"] == 1].sample(n_fraud, random_state=None),
                    full[full["Class"] == 0].sample(n - n_fraud, random_state=None),
                ]).sort_values("timestamp")
                ss.batch_df = part.reset_index(drop=True)
                ss.batch_is_sample = True
                ss.batch_result = None
    else:
        template = pd.DataFrame([{
            "customer_id": "CUST00001", "amount": 1500, "currency": "INR",
            "timestamp": "2026-11-21 14:00:00", "merchant_id": "", "payee_id": "",
            "device_id": "DEV00001", "transaction_type": "UPI", "card_present": 0,
            "is_international": 0, "transaction_latitude": 19.07, "transaction_longitude": 72.87,
            "label": "",
        }])
        st.download_button(
            "⬇️ Download CSV template", template.to_csv(index=False), "fraudshield_template.csv", "text/csv"
        )
        st.caption(
            f"Required: `{', '.join(BATCH_REQUIRED)}`. Optional: `{', '.join(BATCH_OPTIONAL)}`. "
            "Add a **`label`** column (1 = fraud, 0 = legit) if you already know the outcome — "
            "labelled rows are saved as reviewed and can be used straight away for retraining."
        )
        up = st.file_uploader("Upload transactions CSV", type=["csv"])
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                missing = [c for c in BATCH_REQUIRED if c not in df_up.columns]
                if missing:
                    st.error(f"Missing required column(s): {missing}")
                else:
                    if "timestamp" in df_up.columns:
                        df_up["timestamp"] = pd.to_datetime(df_up["timestamp"], errors="coerce")
                    ss.batch_df = df_up
                    ss.batch_is_sample = False
                    ss.batch_result = None
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")

    batch_df = ss.get("batch_df")
    if batch_df is not None:
        st.caption(f"{len(batch_df):,} transactions loaded")
        with st.expander("Preview input"):
            st.dataframe(batch_df.head(50), width="stretch")
        is_sample = ss.get("batch_is_sample", False)
        save_batch = st.checkbox(
            "Save results to the database",
            value=not is_sample, disabled=is_sample, key="batch_save",
            help="Saved rows become customer history and can be reviewed / used for retraining.",
        )
        if st.button("🚀 Score all", type="primary"):
            ss.batch_result = run_batch(batch_df, persist=bool(save_batch and not is_sample))

    result = ss.get("batch_result")
    if result is not None:
        st.divider()
        n_rows, n_flag = len(result), int(result["predicted_fraud"].sum())
        m = st.columns(4)
        m[0].metric("Scored", f"{n_rows:,}")
        m[1].metric("Flagged as fraud", f"{n_flag:,}")
        m[2].metric("Flag rate", f"{n_flag / n_rows:.1%}")
        m[3].metric("Total flagged value", money(result.loc[result["predicted_fraud"], "amount"].sum()))

        truth = "Class" if "Class" in result.columns else ("label" if "label" in result.columns else None)
        if truth and result[truth].notna().any():
            known = result[result[truth].notna()]
            y = known[truth].astype(int)
            p = known["predicted_fraud"].astype(int)
            tp = int(((y == 1) & (p == 1)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            tn = int(((y == 0) & (p == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            st.markdown("##### Accuracy against the known labels")
            q = st.columns(6)
            q[0].metric("Precision", f"{prec:.1%}")
            q[1].metric("Recall", f"{rec:.1%}")
            q[2].metric("True positives", tp)
            q[3].metric("False positives", fp)
            q[4].metric("False negatives", fn)
            q[5].metric("True negatives", tn)

        show_cols = [c for c in [
            "transaction_id_api", "customer_id", "timestamp", "amount", "currency", "transaction_type",
            "Class", "label", "fraud_type", "fraud_probability", "predicted_fraud", "risk_level",
        ] if c in result.columns]
        view = result[show_cols].sort_values("fraud_probability", ascending=False)
        st.dataframe(
            view, hide_index=True, width="stretch",
            column_config={
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "amount": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button(
            "⬇️ Download results", view.to_csv(index=False), "fraudshield_results.csv", "text/csv"
        )


# =====================================================
# TAB 3 - SAVED TRANSACTIONS (SQLite) + REVIEW
# =====================================================

with tab_txns:
    try:
        stats = api("GET", "/db/stats")
    except ApiError as exc:
        st.error(str(exc))
        stats = None

    if stats:
        k = st.columns(5)
        k[0].metric("Saved", f"{stats['total']:,}")
        k[1].metric("Flagged", f"{stats['flagged']:,}")
        k[2].metric("Awaiting review (flagged)", f"{stats['flagged_unreviewed']:,}")
        k[3].metric("Reviewed", f"{stats['reviewed']:,}")
        k[4].metric("AI briefings", f"{stats['with_ai']:,}")

        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        flag_f = f1.selectbox("Model verdict", ["All", "Flagged", "Not flagged"], key="txn_f_flag")
        rev_f = f2.selectbox(
            "Your review", ["Any", "Unreviewed", "Reviewed", "Confirmed fraud", "Confirmed legit"], key="txn_f_rev"
        )
        cust_f = f3.text_input("Customer contains", key="txn_f_cust")
        lim_f = f4.selectbox("Show", [50, 100, 200, 500], index=1, key="txn_f_lim")

        params: dict[str, Any] = {
            "review": {"Any": "any", "Unreviewed": "unreviewed", "Reviewed": "reviewed",
                       "Confirmed fraud": "fraud", "Confirmed legit": "legit"}[rev_f],
            "limit": lim_f,
        }
        if flag_f != "All":
            params["flagged"] = "true" if flag_f == "Flagged" else "false"
        if cust_f.strip():
            params["customer"] = cust_f.strip()

        try:
            listing = api("GET", "/transactions", params=params)
        except ApiError as exc:
            st.error(str(exc))
            listing = {"total": 0, "transactions": []}

        rows = listing["transactions"]
        if not rows:
            st.info("No saved transactions match. Check a transaction with **Save to database** ticked, "
                    "or upload a CSV in *Batch Scoring*.")
        else:
            st.caption(f"Showing {len(rows)} of {listing['total']:,}")
            tdf = pd.DataFrame(rows)
            tdf["review"] = tdf["review_label"].apply(review_text)
            st.dataframe(
                tdf[["transaction_id", "timestamp", "customer_id", "amount", "currency",
                     "transaction_type", "fraud_probability", "risk_level", "predicted_fraud",
                     "review", "has_ai", "model_version"]],
                hide_index=True, width="stretch",
                column_config={
                    "fraud_probability": st.column_config.ProgressColumn(
                        "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f"),
                    "amount": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            labels = {
                r["transaction_id"]:
                    f"{r['transaction_id']} · {r['customer_id']} · {r['amount']:,.0f} {r['currency']} · "
                    f"{r['fraud_probability']:.0%} · {review_text(r['review_label'])}"
                for r in rows
            }
            chosen = st.selectbox("Open a transaction", list(labels), format_func=labels.get, key="txn_open")

            try:
                det = api("GET", f"/transactions/{chosen}")
            except ApiError as exc:
                st.error(str(exc))
                det = None

            if det:
                st.divider()
                render_assessment({
                    "is_fraud": det["predicted_fraud"], "risk_level": det["risk_level"],
                    "fraud_probability": det["fraud_probability"], "threshold": det["threshold"],
                    "recommended_action": det["recommended_action"], "transaction_id": det["transaction_id"],
                })
                info = st.columns(4)
                info[0].metric("Customer", det["customer_id"])
                info[1].metric("Amount", money(det["amount"], det["currency"]))
                info[2].metric("When", str(det["timestamp"])[:16])
                info[3].metric("Scored by", det.get("model_version") or "—")

                render_reasons(det["reasons"], None, det["features"])
                st.divider()
                left, right = st.columns(2)
                with left:
                    render_ai(det["transaction_id"], det.get("ai_explanation"), det.get("ai_model"),
                              None, det["predicted_fraud"], "txn")
                with right:
                    render_review(det["transaction_id"], det.get("review_label"),
                                  det.get("review_note"), det["predicted_fraud"], "txn")

                with st.expander("🗑️ Delete this transaction"):
                    st.caption("Removes it from the database **and** from the customer's history.")
                    sure = st.checkbox("I'm sure", key=f"txn_del_ok_{chosen}")
                    if st.button("Delete", disabled=not sure, key=f"txn_del_{chosen}"):
                        try:
                            api("DELETE", f"/transactions/{chosen}")
                            st.cache_data.clear()
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))


# =====================================================
# TAB 5 - DASHBOARD
# =====================================================

with tab_dash:
    try:
        a = get("/analytics/summary")
    except ApiError as exc:
        st.error(str(exc))
        a = {}

    if a.get("database"):
        dbs = a["database"]
        st.markdown("##### Your saved transactions")
        k = st.columns(6)
        k[0].metric("Saved", f"{dbs['total']:,}")
        k[1].metric("Flagged", f"{dbs['flagged']:,}")
        k[2].metric("Reviewed", f"{dbs['reviewed']:,}")
        k[3].metric("Confirmed fraud", f"{dbs['confirmed_fraud']:,}")
        k[4].metric("Model false alarms", f"{dbs['false_positives']:,}",
                    help="Flagged by the model but you confirmed legit")
        k[5].metric("Missed fraud", f"{dbs['missed_fraud']:,}",
                    help="Not flagged by the model but you confirmed fraud")
        st.divider()

    if "total_transactions" in a:
        st.markdown("##### Original dataset")
        k = st.columns(5)
        k[0].metric("Transactions", f"{a['total_transactions']:,}")
        k[1].metric("Fraud cases", f"{a['fraud_transactions']:,}")
        k[2].metric("Fraud rate", f"{a['fraud_rate_pct']}%")
        k[3].metric("Customers", f"{a['customers']:,}")
        k[4].metric("INR lost to fraud", f"₹{a['inr_fraud_amount']:,.0f}")
        st.caption(f"Dataset period: {a['date_from'][:10]} → {a['date_to'][:10]}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Fraud by type")
            ft = pd.DataFrame(a["by_fraud_type"])
            if not ft.empty:
                st.altair_chart(
                    alt.Chart(ft).mark_bar(color=RED).encode(
                        x=alt.X("fraud_type:N", title=None, sort="-y"),
                        y=alt.Y("cases:Q", title="Fraud cases"),
                        tooltip=["fraud_type", "cases"],
                    ).properties(height=280),
                    width="stretch",
                )
        with c2:
            st.markdown("##### Fraud rate by payment channel")
            bt = pd.DataFrame(a["by_transaction_type"])
            st.altair_chart(
                alt.Chart(bt).mark_bar(color="#06B6D4").encode(
                    x=alt.X("transaction_type:N", title=None, sort="-y"),
                    y=alt.Y("fraud_rate:Q", title="Fraud rate (%)"),
                    tooltip=["transaction_type", "transactions", "fraud", "fraud_rate"],
                ).properties(height=280),
                width="stretch",
            )

        st.markdown("##### Daily fraud cases")
        daily = pd.DataFrame(a["daily"])
        daily["day"] = pd.to_datetime(daily["day"])
        st.altair_chart(
            alt.Chart(daily).mark_area(line={"color": RED}, color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="rgba(239,68,68,0.05)", offset=0),
                       alt.GradientStop(color="rgba(239,68,68,0.45)", offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            )).encode(
                x=alt.X("day:T", title=None),
                y=alt.Y("fraud:Q", title="Fraud cases"),
                tooltip=["day:T", "transactions", "fraud", "fraud_rate"],
            ).properties(height=260),
            width="stretch",
        )

        st.markdown("##### By currency")
        st.dataframe(pd.DataFrame(a["by_currency"]), hide_index=True, width="stretch")
    else:
        st.info("Analytics unavailable — the history file (data/processed_data.csv) was not loaded.")

    st.markdown("##### Recently scored in this API session")
    try:
        rp = api("GET", "/predictions/recent", params={"limit": 50})["predictions"]
    except ApiError:
        rp = []
    if rp:
        st.dataframe(
            pd.DataFrame(rp), hide_index=True, width="stretch",
            column_config={
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud probability", min_value=0.0, max_value=1.0, format="%.3f")
            },
        )
    else:
        st.caption("Nothing scored yet — try the *Check Transaction* tab.")


# =====================================================
# TAB 6 - MODEL INSIGHTS
# =====================================================


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


with tab_model:
    try:
        info = get("/model/info")
    except ApiError as exc:
        st.error(str(exc))
        info = None

    if info:
        st.markdown(f"#### {info['model_name']}  ·  `{info.get('model_version')}`")
        tm = info.get("test_metrics")
        if tm:
            st.caption("Metrics of the original model from `modules/model.py` (see the *Retrain* tab "
                       "for the metrics of retrained versions).")
            m = st.columns(6)
            m[0].metric("Precision", f"{tm['precision']:.3f}")
            m[1].metric("Recall", f"{tm['recall']:.3f}")
            m[2].metric("F1", f"{tm['f1_score']:.3f}")
            m[3].metric("PR-AUC", f"{tm['pr_auc']:.4f}")
            m[4].metric("False positives", int(tm["false_positives"]))
            m[5].metric("False negatives", int(tm["false_negatives"]))

        if info.get("model_comparison"):
            st.markdown("##### Model comparison")
            st.dataframe(pd.DataFrame(info["model_comparison"]), hide_index=True, width="stretch")

        if info.get("feature_importance"):
            st.markdown("##### Global feature importance (mean |SHAP|, original model)")
            fi = pd.DataFrame(info["feature_importance"]).head(15)
            st.altair_chart(
                alt.Chart(fi).mark_bar(color="#A78BFA").encode(
                    x=alt.X("mean_absolute_shap:Q", title="Mean |SHAP value|"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", alt.Tooltip("mean_absolute_shap:Q", format=".4f")],
                ).properties(height=380),
                width="stretch",
            )

        st.markdown("##### Evaluation & explainability charts")
        imgs = [
            (REPORT_DIR / "final_confusion_matrix.png", "Confusion matrix (test set)"),
            (REPORT_DIR / "final_precision_recall.png", "Precision-recall curve"),
            (EXPLAIN_DIR / "shap_summary.png", "SHAP summary"),
            (EXPLAIN_DIR / "fraud_waterfall.png", "Example: explaining a fraud case"),
            (EXPLAIN_DIR / "legitimate_waterfall.png", "Example: explaining a legitimate case"),
            (INTEL_DIR / "ablation_reports" / "ablation_average_precision.png", "Basic vs personalized features"),
        ]
        existing = [i for i in imgs if i[0].exists()]
        for i in range(0, len(existing), 2):
            cols = st.columns(2)
            for col, (path, cap) in zip(cols, existing[i:i + 2]):
                with col:
                    show_image(path, cap)

        with st.expander("Fraud intelligence charts"):
            intel = sorted(INTEL_DIR.glob("*.png")) + sorted(ANALYSIS_DIR.glob("*.png"))
            for i in range(0, len(intel), 2):
                cols = st.columns(2)
                for col, path in zip(cols, intel[i:i + 2]):
                    with col:
                        show_image(path, path.stem.replace("_", " ").title())

        with st.expander(f"Model input features ({len(info['features'])})"):
            st.write(", ".join(info["features"]))


# =====================================================
# TAB 7 - CUSTOMER 360
# =====================================================

with tab_customer:
    try:
        ids = get("/customers", limit=5000)["customers"]
    except ApiError:
        ids = []
    if ids:
        cid = st.selectbox("Customer", ids, key="c360_customer")
        try:
            p = get(f"/customers/{cid}", recent=25)
        except ApiError as exc:
            st.error(str(exc))
            p = None
        if p:
            m = st.columns(5)
            m[0].metric("Transactions", p["transaction_count"])
            m[1].metric("Failed", p["failed_count"])
            m[2].metric("Avg amount", money(p["avg_amount"], p["primary_currency"]))
            m[3].metric("Max amount", money(p["max_amount"], p["primary_currency"]))
            m[4].metric("Account created", (str(p["account_creation_date"]) if p["account_creation_date"] else "—")[:10])

            m = st.columns(4)
            m[0].metric("Known merchants", p["known_merchants"])
            m[1].metric("Known payees", p["known_payees"])
            m[2].metric("Usual hours", ", ".join(f"{h:02d}" for h in p["usual_hours"][:6]) + ("…" if len(p["usual_hours"]) > 6 else ""))
            m[3].metric("Last seen", (p["last_transaction_at"] or "—")[:16])

            st.markdown("##### Home location")
            st.map(
                pd.DataFrame({"lat": [p["registered_latitude"]], "lon": [p["registered_longitude"]]}),
                zoom=8, height=240,
            )
            st.markdown("##### Recent transactions")
            st.dataframe(pd.DataFrame(p["recent_transactions"]), hide_index=True, width="stretch")


# =====================================================
# TAB 4 - RETRAIN  (kept last: it may poll in a loop while a job runs)
# =====================================================


def metric_rows(m: dict, label: str) -> dict:
    return {
        "Evaluated on": label,
        "Rows": m["rows"],
        "Fraud rows": m["fraud_rows"],
        "Precision": round(m["precision"], 4),
        "Recall": round(m["recall"], 4),
        "F1": round(m["f1_score"], 4),
        "PR-AUC": None if m["pr_auc"] is None else round(m["pr_auc"], 5),
        "False positives": m["false_positives"],
        "False negatives": m["false_negatives"],
        "Threshold": round(m["threshold"], 4),
    }


def cb_activate(version: str) -> None:
    try:
        api("POST", f"/models/versions/{version}/activate")
        st.cache_data.clear()
        flash("retrain", "success", f"Model {version} is now active.")
    except ApiError as exc:
        flash("retrain", "error", str(exc))


with tab_retrain:
    try:
        rs = api("GET", "/retrain/status")
        versions = api("GET", "/models/versions")
    except ApiError as exc:
        st.error(str(exc))
        rs, versions = None, None

    if rs:
        data, jobinfo = rs["data"], rs["job"]
        running = jobinfo["state"] == "running"

        st.markdown(f"#### Active model: **{rs['model_name']}** · `{rs['active_version']}`")
        show_flash("retrain")

        k = st.columns(5)
        k[0].metric("Reviewed transactions", data["reviewed"])
        k[1].metric("Confirmed fraud", data["confirmed_fraud"])
        k[2].metric("Confirmed legit", data["confirmed_legit"])
        k[3].metric("New since last retrain", data["labeled_new"])
        k[4].metric("Flagged, not reviewed", data["flagged_unreviewed"],
                    help="Review these in the Transactions tab to add them to the training data")

        with st.expander("How retraining works", expanded=False):
            st.markdown(
                """
- Training data = the **original dataset** + every transaction you **reviewed** as fraud or legit.
- Unreviewed transactions are **never** used: the model's own guesses are not ground truth.
- A quarter of your reviewed rows is held out so you can see how the old and new models do on data they never trained on.
- Your **original test set stays untouched** as a fixed benchmark. A new model is only *recommended* if it does not get worse there.
- A retrained model is saved as a **new version** and does **not** replace the active one until you activate it. You can switch back to any earlier version at any time.
- With only a handful of reviewed rows the change will be tiny; the more (and the more varied) your reviews, the more the model can adapt.
                """
            )

        c1, c2, c3 = st.columns(3)
        weight = c1.slider("Weight of each reviewed row", 1.0, 20.0, 3.0, step=0.5,
                           help="Your reviewed rows are few compared with the 40,000 original rows, "
                                "so each one can count as several training rows.")
        min_rows = c2.number_input("Minimum reviewed rows", 1, 1000, 10)
        auto_act = c3.checkbox("Activate automatically if there is no regression", value=False)

        enough = data["reviewed"] >= min_rows
        if not enough:
            st.warning(f"You have {data['reviewed']} reviewed transaction(s); at least {min_rows} are needed. "
                       "Review more in the **Transactions** tab, or upload a CSV with a `label` column.")
        if data["reviewed"] and data["confirmed_fraud"] == 0:
            st.info("None of your reviewed transactions are fraud yet — the model can still learn from "
                    "legitimate examples, but fraud examples add the most.")

        if st.button("🔁 Retrain model", type="primary", disabled=(not enough) or running, width="stretch"):
            try:
                api("POST", "/retrain", json={
                    "new_row_weight": float(weight), "min_new_rows": int(min_rows),
                    "auto_activate": bool(auto_act),
                })
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

        # ---- live progress while a job runs
        if running:
            box = st.empty()
            for _ in range(900):
                try:
                    j = api("GET", "/retrain/status")["job"]
                except ApiError as exc:
                    box.error(str(exc))
                    break
                with box.container():
                    st.progress(min(max(j["progress"], 0.0), 1.0), text=j["message"] or "Working…")
                    st.caption(" → ".join(x.split("  ", 1)[-1] for x in j["log"][-4:]))
                if j["state"] != "running":
                    break
                time.sleep(1.0)
            st.cache_data.clear()
            st.rerun()

        # ---- result of the last job
        if jobinfo["state"] == "failed":
            st.error(f"Retraining failed: {jobinfo['error']}")
        elif jobinfo["state"] == "completed" and jobinfo["result"]:
            res = jobinfo["result"]
            st.divider()
            st.markdown(f"#### Result: `{res['version']}`")
            (st.success if res["recommended"] else st.warning)(res["recommendation_reason"])
            if res.get("activated"):
                st.success(f"✅ {res['version']} was activated automatically.")

            c = st.columns(4)
            c[0].metric("Original rows", f"{res['n_original']:,}")
            c[1].metric("Reviewed rows used", res["n_new"])
            c[2].metric("…of which held out", res["n_new_holdout"])
            c[3].metric("New threshold", f"{res['threshold']:.3f}")

            mt = res["metrics"]
            table = [
                {"Model": "Previous (active at training time)", **metric_rows(mt["old_on_test"], "Original test set")},
                {"Model": f"New ({res['version']})", **metric_rows(mt["new_on_test"], "Original test set")},
            ]
            if "new_on_new_holdout" in mt:
                table += [
                    {"Model": "Previous (active at training time)", **metric_rows(mt["old_on_new_holdout"], "Your held-out reviews")},
                    {"Model": f"New ({res['version']})", **metric_rows(mt["new_on_new_holdout"], "Your held-out reviews")},
                ]
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            if "new_on_new_holdout" in mt and mt["new_on_new_holdout"]["rows"] < 30:
                st.caption("⚠️ Your held-out reviews are very few, so those numbers are only indicative.")

            if rs["active_version"] != res["version"]:
                st.button(f"✅ Activate {res['version']}", on_click=cb_activate, args=(res["version"],),
                          key=f"act_res_{res['version']}")

        # ---- version registry
        if versions:
            st.divider()
            st.markdown("#### Model versions")
            vdf = pd.DataFrame(versions["versions"])
            vdf["active"] = vdf["is_active"].map({True: "● active", False: ""})
            st.dataframe(
                vdf[["version", "active", "source", "model_name", "created_at", "threshold", "n_new", "new_row_weight", "notes"]],
                hide_index=True, width="stretch",
            )
            inactive = [v["version"] for v in versions["versions"] if not v["is_active"]]
            if inactive:
                a1, a2 = st.columns([3, 1])
                pick_v = a1.selectbox("Switch to version", inactive, key="ver_pick")
                a2.write("")
                a2.write("")
                a2.button("Activate", on_click=cb_activate, args=(pick_v,), key="ver_activate", width="stretch")