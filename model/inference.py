"""
inference.py — the scoring seam. The ONLY entry point the API and UI may call.

    score(...) -> ScoreResult(probability, risk_band, risk_factors, ...)

SCOPE
-----
This service answers ONE question: how likely is this transaction to be fraud,
and why. It deliberately does NOT decide approve / review / block.

Decisioning, triage bands, and compliance rules live in a separate system. That
separation is a good one: compliance constraints change on a different cadence
than models, they need to be auditable by non-engineers, and they must keep
working if this service is unavailable. Coupling them to the model service ties
together two things that should fail independently.

What the consuming system receives is a calibrated probability plus the factors
that drove it. What it does with that is its own concern.

WHY CALIBRATION MATTERS HERE SPECIFICALLY
-----------------------------------------
Because a downstream system will threshold on these numbers, they must mean what
they say. scale_pos_weight inflates raw model scores, so an uncalibrated 0.6 is
NOT a 60% fraud probability. The isotonic calibrator fitted during training
corrects that. Never expose raw scores to a consumer that bands on them.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path as _P

import numpy as np

logger = logging.getLogger(__name__)
ROOT = _P(__file__).resolve().parent.parent


# ── display configuration ────────────────────────────────────────────────────
# Words, not decimals: a non-technical reader parses "High" instantly. These are
# ADVISORY display labels only -- they carry no decisioning authority.
RISK_BAND_LABELS = [(0.85, "Critical"), (0.50, "High"), (0.15, "Elevated"), (0.0, "Low")]

# Maps model feature names to language a reviewer or customer can read.
# Business-editable: add entries as features are added.
FEATURE_LABELS = {
    "send_amount": "Transfer amount",
    "cents": "Amount pattern (round or unusual cents)",
    "transfer_fee": "Transfer fee",
    "rate": "Exchange rate",
    "sending_from": "Origin country",
    "sending_to": "Destination country",
    "sending_currency": "Sending currency",
    "receiving_currency": "Receiving currency",
    "delivery_method": "Payout method",
    "payment_method": "Funding method",
    "paymentProvider": "Payment provider",
    "source": "Channel (app or web)",
    "description_code": "Stated purpose of transfer",
    "network_name": "Card network",
    "avs_code": "Address verification result",
    "verifiedStatus3Ds": "3-D Secure result",
    "isChallenged3Ds": "3-D Secure challenge issued",
    "threeds_reported": "3-D Secure reported by provider",
    "ip_country_mismatch": "Sign-in location differs from stated origin",
    "ipDetails__security__vpn": "VPN detected",
    "ipDetails__security__proxy": "Proxy detected",
    "ipDetails__security__tor": "Tor network detected",
    "ipDetails__security__relay": "Relay detected",
    "ipDetails__location__country_code": "Sign-in country",
    "ipDetails__location__city": "Sign-in city",
    "ipDetails__network__autonomous_system_number": "Network provider",
    "sender_email_domain": "Sender email provider",
    "recipient_email_domain": "Recipient email provider",
    "hour": "Time of day",
    "weekday": "Day of week",
    "user_age_days": "Account age",
    "user_age_years": "Customer age",
    "recipient_age_days": "Recipient age (since added)",
    "is_first_send_to_recipient": "First transfer to this recipient",
    "prior_sends_to_recipient": "Previous transfers to this recipient",
    "is_recipient_changed": "Recipient changed late in the flow",
    "limit_utilisation": "Proportion of sending limit used",
    "billing_country_match": "Billing country matches account country",
    "user_threshold_level_no": "Account verification level",
    "user_phone_carrier_type": "Phone line type",
    "user_isVerified": "Identity verified",
    "user_isDocumentExpired": "Identity document expired",
    "user_isPinLocked": "Account PIN locked",
    "user_addressChangeCounter": "Recent address changes",
    "user_cardDeletionCounter": "Recent card removals",
    "user_country": "Account country",
    "user_signup_country": "Sign-up country",
    "user_occupation": "Stated occupation",
    "user_isBusinessAccount": "Business account",
    "recipient_country": "Recipient country",
    "recipient_relationship": "Stated relationship to recipient",
    "recipient_account_type": "Recipient account type",
    "recipient_doc_bank_name": "Recipient bank",
    "recipient_bank_name": "Recipient bank",
    "recipient_payoutPartner": "Payout partner",
    "recipient_status": "Recipient account status",
    "user_id_txn_1h": "Transfers in the past hour",
    "user_id_txn_24h": "Transfers in the past 24 hours",
    "user_id_txn_168h": "Transfers in the past week",
    "user_id_amt_1h": "Amount sent in the past hour",
    "user_id_amt_24h": "Amount sent in the past 24 hours",
    "user_id_amt_168h": "Amount sent in the past week",
}

# Feature-name prefixes with no published business meaning. Used to separate
# "readable" drivers from "behavioural" ones in explanations.
OPAQUE_PREFIXES = ("V", "C", "D", "M", "id_")


# ── result ───────────────────────────────────────────────────────────────────
@dataclass
class ScoreResult:
    """What this service returns. No decision field, by design."""
    probability: float
    risk_factors: list[dict] = field(default_factory=list)
    behavioural_factors: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def risk_band(self) -> str:
        """Advisory display label. NOT a decision."""
        for cut, name in RISK_BAND_LABELS:
            if self.probability >= cut:
                return name
        return "Low"

    def to_dict(self) -> dict:
        return {
            "probability": round(float(self.probability), 6),
            "risk_band": self.risk_band,
            "risk_factors": self.risk_factors,
            "behavioural_factors": self.behavioural_factors,
            "latency_ms": round(float(self.latency_ms), 2),
        }


# ── scoring ──────────────────────────────────────────────────────────────────
def predict_calibrated(model, calibrator, X):
    """Raw features -> calibrated fraud probability.

    Always route consumer-facing scores through the calibrator: downstream
    systems band on these values, so they must be true probabilities."""
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw)


def score(model, calibrator, X_row, txn: dict = None, booster=None,
          feature_order=None, k: int = 3, explain_factors: bool = True) -> ScoreResult:
    """Score ONE transaction and, optionally, explain it.

    X_row must already be through ModelPrep.transform, using the SAME fitted
    preprocessor as training, or the features will not line up.
    """
    import time
    t0 = time.perf_counter()
    prob = float(predict_calibrated(model, calibrator, X_row)[0])

    interp, behav = [], []
    if explain_factors and booster is not None and feature_order is not None:
        try:
            interp = top_risk_factors(booster, X_row, feature_order, txn or {},
                                      k=k, interpretable_only=True)
            behav = top_risk_factors(booster, X_row, feature_order, txn or {},
                                     k=k, interpretable_only=False)
        except Exception as e:                     # explanation must never break scoring
            logger.warning(f"explanation unavailable: {e}")

    return ScoreResult(probability=prob, risk_factors=interp,
                       behavioural_factors=behav,
                       latency_ms=(time.perf_counter() - t0) * 1000)


def score_batch(model, calibrator, X) -> np.ndarray:
    """Calibrated probabilities for many rows. No explanations (SHAP per row is
    far too slow for bulk scoring; explain individual cases on demand)."""
    return predict_calibrated(model, calibrator, X)


# ── explanations ─────────────────────────────────────────────────────────────
def label_for(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.endswith("_FE"):
        base = feature[:-3]
        return f"{FEATURE_LABELS.get(base, base)} (how common this value is)"
    if feature.startswith(OPAQUE_PREFIXES):
        return f"Behavioural signal {feature}"
    return feature.replace("_", " ").capitalize()


def is_interpretable(feature: str) -> bool:
    if feature in FEATURE_LABELS:
        return True
    if feature.endswith("_FE"):
        return feature[:-3] in FEATURE_LABELS
    return not feature.startswith(OPAQUE_PREFIXES)


def top_risk_factors(booster, row_df, feature_order, txn: dict, k: int = 3,
                     interpretable_only: bool = True) -> list[dict]:
    """Top-k features that pushed THIS transaction's risk up.

    Uses exact tree SHAP contributions (xgboost pred_contribs), which sum to the
    raw log-odds. They therefore explain the RAW score, not the calibrated
    probability -- present them as directional drivers ("these raised the risk"),
    never as a percentage breakdown of the final number.

    `interpretable_only=True` filters to features with business meaning. The model
    still uses the rest; `hidden_signal_weight` records how much risk came from
    them, so the omission is visible rather than silent.
    """
    import xgboost as xgb

    dm = xgb.DMatrix(row_df[feature_order], enable_categorical=True)
    contribs = booster.predict(dm, pred_contribs=True)[0]
    values = contribs[:-1]                         # drop the bias term

    out, hidden = [], 0.0
    for i in np.argsort(values)[::-1]:
        if values[i] <= 0:
            break                                  # sorted: no positive drivers left
        feat = feature_order[i]
        if interpretable_only and not is_interpretable(feat):
            hidden += float(values[i])
            continue
        out.append({"feature": feat, "label": label_for(feat),
                    "value": _clean(txn.get(feat)), "impact": float(values[i])})
        if len(out) == k:
            break
    for f in out:
        f["hidden_signal_weight"] = hidden
    return out


def _clean(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def explain(result: ScoreResult) -> str:
    """Plain-English account of a score, for reviewers and support staff.

    Describes RISK, not a decision: the decision is made elsewhere."""
    lines = [f"{result.risk_band} risk ({result.probability:.1%})"]
    if result.risk_factors:
        lines.append("Main contributing factors:")
        for i, f in enumerate(result.risk_factors, 1):
            val = f.get("value")
            lines.append(f"  {i}. {f['label']}" + (f" — {val}" if val is not None else ""))
        if result.risk_factors[0].get("hidden_signal_weight", 0) > 0:
            lines.append("  (Additional behavioural signals also contributed.)")
    return "\n".join(lines)


# ── evaluation helper ────────────────────────────────────────────────────────
def band_distribution(probs, y_true=None, bands=None):
    """How scores spread across risk bands, and the fraud rate in each.

    ANALYSIS ONLY -- for judging whether the model separates risk usefully, and
    for telling the consuming team what to expect at each band. It does not
    prescribe thresholds; that system owns its own banding."""
    import pandas as pd
    bands = bands or RISK_BAND_LABELS
    probs = np.asarray(probs, dtype=float)

    names = []
    for p in probs:
        for cut, name in bands:
            if p >= cut:
                names.append(name)
                break
        else:
            names.append("Low")

    df = pd.DataFrame({"band": names})
    if y_true is not None:
        df["label"] = np.asarray(y_true)

    rows = []
    for _, name in bands:
        m = df["band"] == name
        row = {"band": name, "count": int(m.sum()), "pct": float(m.mean() * 100)}
        if y_true is not None and m.sum():
            row["fraud_rate_pct"] = float(df.loc[m, "label"].mean() * 100)
            row["fraud_caught"] = int(df.loc[m, "label"].sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")
