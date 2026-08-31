"""
aggressiveness.py — how readily the system flags, and what a false positive costs.

TWO SEPARATE THINGS, DELIBERATELY
---------------------------------
    PRESET   a policy dial. "How cautious is the business willing to be right
             now?" A judgement call that belongs to a human, adjustable without
             touching code or retraining anything.

    COST     a measurement. "What does THIS false positive actually cost?"
             Grounded in transfer_fee, send_amount, and churn risk rather than a
             single guessed constant.

Multiplied together they give a threshold that is both business-controlled and
data-grounded. Neither alone is enough: presets on their own are arbitrary, and
per-transaction costs on their own leave no lever for a human to pull.

WHY THE OLD FLAT $25 WAS THE WEAK POINT
---------------------------------------
The false-negative side was already grounded in real per-transaction data (the
amount actually lost). The false-positive side was one constant applied to every
transaction. That asymmetry is what this module fixes: a wrongly-blocked $20
top-up and a wrongly-blocked $4,000 school-fees transfer are not the same
mistake, and the second customer is far more likely to leave.

A false positive is really three costs:
    ops     support handling and manual review. Genuinely ~fixed per incident.
    revenue the fee you would have earned. NOT fixed -- and already in the data
            as transfer_fee, sitting unused.
    churn   P(customer leaves | wrongly blocked) x their forward value. Usually
            the DOMINANT term, and the one everyone hand-waves.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── the policy dial ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Preset:
    name: str
    multiplier: float
    summary: str


# Named by AGGRESSIVENESS OF DETECTION, not by cost level.
#
# "very aggressive" unambiguously means "flags more". Naming these by the FP cost
# instead (low/medium/high) reads backwards: someone picking "high" expecting
# maximum strictness would get the opposite, because a HIGH false-positive cost
# makes the system flag LESS.
#
# Mechanically the multiplier scales the measured FP cost. Cheaper false
# positives -> lower optimal threshold -> more flags.
PRESETS = {
    "very_aggressive": Preset(
        "very_aggressive", 0.25,
        "Catch as much fraud as possible; accept many false positives. "
        "Fraud spike, or a corridor under active attack."),
    "aggressive": Preset(
        "aggressive", 0.5,
        "Lean towards catching fraud. Elevated risk periods."),
    "balanced": Preset(
        "balanced", 1.0,
        "Use the measured cost as-is. The default."),
    "cautious": Preset(
        "cautious", 2.0,
        "Lean towards not disturbing customers. Growth periods, or after "
        "false-positive complaints."),
    "very_cautious": Preset(
        "very_cautious", 4.0,
        "Almost never block a legitimate customer; accept more missed fraud."),
}

DEFAULT_PRESET = "balanced"
ORDER = ["very_aggressive", "aggressive", "balanced", "cautious", "very_cautious"]


def get_preset(name: str) -> Preset:
    key = (name or DEFAULT_PRESET).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PRESETS:
        raise ValueError(f"unknown preset {name!r}. Options: {ORDER}")
    return PRESETS[key]


# ── the measured cost ────────────────────────────────────────────────────────
@dataclass
class FPCostModel:
    """Per-transaction false-positive cost.

        fp_cost(txn) = ops_cost + lost_fee + P(churn) x forward_value

    ops_cost      support handling per wrongly-blocked customer. A business
                  input; not measurable from transaction data.
    churn_prob    P(customer stops transacting | wrongly blocked). MEASURABLE
                  from your own history -- see estimate_from_history().
    clv_months    months of forward volume a retained customer represents.
    """
    ops_cost: float = 15.0
    churn_prob: float = 0.05
    clv_months: float = 12.0
    monthly_volume_multiple: float = 1.0
    source: str = "defaults"          # "defaults" | "measured"

    def per_transaction(self, df: pd.DataFrame) -> np.ndarray:
        """Cost array aligned to df. Falls back gracefully on missing columns."""
        n = len(df)
        fee = (pd.to_numeric(df["transfer_fee"], errors="coerce").fillna(0.0).to_numpy()
               if "transfer_fee" in df.columns else np.zeros(n))
        amt = (pd.to_numeric(df["send_amount"], errors="coerce").fillna(0.0).to_numpy()
               if "send_amount" in df.columns else np.zeros(n))

        # forward value proxy: this customer's typical transfer, projected over
        # the retention horizon. Deliberately simple -- a per-user CLV lookup
        # would be better once one exists.
        forward_value = amt * self.monthly_volume_multiple * self.clv_months
        return self.ops_cost + fee + self.churn_prob * forward_value

    def describe(self) -> dict:
        return {"ops_cost": self.ops_cost, "churn_prob": self.churn_prob,
                "clv_months": self.clv_months,
                "monthly_volume_multiple": self.monthly_volume_multiple,
                "source": self.source}


def build_fp_costs(df: pd.DataFrame, preset: str = DEFAULT_PRESET,
                   model: FPCostModel = None) -> np.ndarray:
    """Per-transaction FP cost, scaled by the chosen aggressiveness preset."""
    model = model or FPCostModel()
    return model.per_transaction(df) * get_preset(preset).multiplier


# ── the "auto" setting: measure the parameters from history ──────────────────
def estimate_from_history(training_set: pd.DataFrame,
                          ops_cost: float = 15.0,
                          horizon_days: int = 90) -> FPCostModel:
    """Derive churn probability and forward value from the company's own data.

    THE MEASUREMENT THAT MATTERS
    ----------------------------
    label_source == "confirmed_legit_reviewed" identifies transactions that were
    flagged and then cleared on review: CONFIRMED WRONGFUL BLOCKS, with dates.
    Comparing what those customers did next against everyone else turns the
    biggest guess in the cost model into a measurement.

    This is a naive before/after comparison, not a controlled study: customers
    who get flagged differ systematically from those who do not (higher amounts,
    newer accounts), so some of the gap is selection rather than causation. Treat
    the output as a better-informed estimate, not a causal effect. A matched
    comparison on tenure and prior volume would tighten it.
    """
    df = training_set
    needed = {"label_source", "date", "send_amount"}
    if not needed.issubset(df.columns):
        logger.warning(f"cannot estimate: missing {needed - set(df.columns)}. "
                       f"Using defaults.")
        return FPCostModel(ops_cost=ops_cost)

    d = df.copy()
    d["_ts"] = pd.to_datetime(d["date"], errors="coerce", utc=True, format="mixed")
    d["_amt"] = pd.to_numeric(d["send_amount"], errors="coerce").fillna(0.0)

    # ---- forward value: typical monthly spend per active customer -----------
    span_days = max((d["_ts"].max() - d["_ts"].min()).days, 1)
    months = span_days / 30.44
    per_txn = float(d["_amt"].median())
    txns_per_month = len(d) / months if months else 0.0

    # ---- churn: what did wrongly-blocked customers do next? ----------------
    fp_mask = d["label_source"] == "confirmed_legit_reviewed"
    n_fp = int(fp_mask.sum())
    churn_prob = 0.05                       # conservative default
    measured = False

    if n_fp >= 30 and "user_age_days" in d.columns:
        # proxy for identity: user_age_days plus signup country is stable per
        # user across their transactions, and survives the pseudonymisation that
        # removed the actual id.
        key_cols = [c for c in ["user_age_days", "user_signup_country",
                                "user_threshold_level_no"] if c in d.columns]
        if key_cols:
            d["_user"] = d[key_cols].astype(str).agg("|".join, axis=1)

            fp_events = d.loc[fp_mask, ["_user", "_ts"]]
            horizon = pd.Timedelta(days=horizon_days)

            churned = 0
            for user, ts in fp_events.itertuples(index=False):
                later = d[(d["_user"] == user) & (d["_ts"] > ts) &
                          (d["_ts"] <= ts + horizon)]
                if later.empty:
                    churned += 1

            # baseline: how often does ANY customer go quiet over the same
            # window? Without this we would attribute ordinary inactivity to the
            # block.
            sample = d.sample(min(len(d), 500), random_state=42)
            base_quiet = 0
            for user, ts in sample[["_user", "_ts"]].itertuples(index=False):
                later = d[(d["_user"] == user) & (d["_ts"] > ts) &
                          (d["_ts"] <= ts + horizon)]
                if later.empty:
                    base_quiet += 1

            fp_rate = churned / max(n_fp, 1)
            base_rate = base_quiet / max(len(sample), 1)
            excess = max(0.0, fp_rate - base_rate)      # attributable to the block
            churn_prob = float(min(excess, 0.5))        # cap: 50% is implausible
            measured = True

            logger.info(f"[auto] {n_fp} confirmed wrongful blocks; "
                        f"{fp_rate:.1%} went quiet within {horizon_days}d "
                        f"vs {base_rate:.1%} baseline -> "
                        f"excess churn {churn_prob:.1%}")
    else:
        logger.info(f"[auto] only {n_fp} confirmed wrongful blocks "
                    f"(need >=30); using default churn probability")

    model = FPCostModel(
        ops_cost=ops_cost,
        churn_prob=churn_prob,
        clv_months=12.0,
        monthly_volume_multiple=max(txns_per_month / max(d["_user"].nunique(), 1)
                                    if "_user" in d.columns else 1.0, 1.0),
        source="measured" if measured else "defaults",
    )
    logger.info(f"[auto] fp cost model: {model.describe()}")
    logger.info(f"[auto] median transfer ${per_txn:,.0f}; "
                f"example FP cost ${model.ops_cost + model.churn_prob * per_txn * model.clv_months:,.0f}")
    return model


# ── reporting ────────────────────────────────────────────────────────────────
def compare_presets(y_true, y_scores, amounts, df: pd.DataFrame,
                    model: FPCostModel = None, metrics_mod=None) -> pd.DataFrame:
    """Optimal threshold and outcome under each preset.

    This is the table to put in front of the business: one row per policy
    setting, showing what it actually costs in fraud caught and customers
    disturbed. It turns an abstract dial into a decision."""
    if metrics_mod is None:
        import metrics as metrics_mod
    model = model or FPCostModel()
    base = model.per_transaction(df)

    rows = []
    for name in ORDER:
        p = PRESETS[name]
        fp = base * p.multiplier
        t, _, _ = metrics_mod.best_threshold(y_true, y_scores, amounts, fp)
        tm = metrics_mod.threshold_metrics(y_true, y_scores, t)
        cm = metrics_mod.cost(y_true, y_scores, amounts, t, fp)
        rows.append({
            "preset": name, "multiplier": p.multiplier, "threshold": round(t, 3),
            "recall": tm["recall"], "precision": tm["precision"],
            "false_positive_rate": tm["false_positive_rate"],
            "flagged_pct": tm["flag_rate"] * 100,
            "missed_fraud": cm["n_missed_fraud"],
            "false_declines": cm["n_false_declines"],
            "cost_per_million": cm["cost_per_million"],
        })
    return pd.DataFrame(rows).set_index("preset")
