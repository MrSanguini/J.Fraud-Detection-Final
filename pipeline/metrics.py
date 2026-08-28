"""
metrics.py, evaluation harness: ranking metrics, threshold metrics, and a money cost.

Deliberately SCHEMA-AGNOSTIC. Every function takes plain arrays
(y_true, y_scores, amounts, threshold, ...) and knows nothing about column names,
so it survived the move from the Vesta prototype to the remittance pipeline
unchanged. Do not add column knowledge here.

WHY A MONEY COST
----------------
AUROC and PR-AUC measure how well the model RANKS risk. Neither tells you where
to draw the approve/decline line: that is a business choice, and every threshold
trades false negatives against false positives. The cost function converts model
quality into currency so a threshold can be chosen on expected loss, and so a
non-technical audience can compare configurations on one number.

    cost = (missed fraud, valued at the amount actually lost)
         + (wrongly declined customers x a parameterised friction cost)

Both sides are parameters, not constants:

  fp_unit_cost      What one wrongly-declined legitimate customer costs. NOT
                    measurable from the data: support handling, lost margin,
                    and churn. Expose it as a slider rather than asserting one
                    figure, so a sceptic can dial in their own assumption.

  fn_loss_fraction  What fraction of a fraudulent transaction the business
                    actually eats. 1.0 assumes the full principal is lost, which
                    is usual for remittance (funds are disbursed and rarely
                    recovered). Set lower if liability sits elsewhere or recovery
                    is real. This is a business input, not an ML one.

  fn_fixed_cost     Per-fraud overhead independent of amount: chargeback fees,
                    investigation time.

WHY PR-AUC, NOT JUST AUROC
--------------------------
At a ~2% fraud rate a model can post an excellent AUROC while being useless at
the operating point you would actually run. PR-AUC's baseline is the fraud rate
itself, so it is reported alongside for context.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import config

logger = logging.getLogger(__name__)

# Defaults from config.py; every function below still accepts an explicit
# override, since these three are business assumptions, not constants.
FP_UNIT_COST = config.FP_UNIT_COST
FN_LOSS_FRACTION = config.FN_LOSS_FRACTION
FN_FIXED_COST = config.FN_FIXED_COST


# ── ranking metrics (threshold-independent) ──────────────────────────────────
def ranking_metrics(y_true, y_scores) -> dict:
    """AUROC and PR-AUC. Independent of any decision threshold.

    `pr_auc_baseline` is the fraud rate: a random model scores that on PR-AUC,
    so it is the number PR_AUC should be compared against."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    base = float(y_true.mean()) if len(y_true) else float("nan")
    if len(np.unique(y_true)) < 2:
        logger.warning("only one class present; ranking metrics are undefined.")
        return {"AUROC": float("nan"), "PR_AUC": float("nan"), "pr_auc_baseline": base}
    return {
        "AUROC": float(roc_auc_score(y_true, y_scores)),
        "PR_AUC": float(average_precision_score(y_true, y_scores)),
        "pr_auc_baseline": base,
    }


# ── threshold metrics ────────────────────────────────────────────────────────
def threshold_metrics(y_true, y_scores, threshold: float) -> dict:
    """Precision / recall / F1 and the confusion matrix at one cut point.

    Accepts either probabilities (with a threshold) or binary decisions
    (pass threshold=0.5), so rules-only configurations evaluate identically."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_scores, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flag_rate": float((tp + fp) / n) if n else 0.0,
        # of legitimate traffic, the share wrongly stopped -- the customer-facing number
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


# ── the money model ──────────────────────────────────────────────────────────
def cost(y_true, y_scores, amounts, threshold: float,
         fp_unit_cost: float = None, fn_loss_fraction: float = None,
         fn_fixed_cost: float = None) -> dict:
    """Expected loss at one threshold.

    Missed fraud is valued at the amount ACTUALLY lost (amount x fn_loss_fraction,
    plus any fixed per-fraud overhead). Wrongly-declined customers cost a flat
    parameterised amount each.

    `amounts` must be the realised loss figure (amount_paid), not the attempted
    amount, and must NOT be a model feature: it is unknown at scoring time.
    """
    fp_unit_cost = FP_UNIT_COST if fp_unit_cost is None else fp_unit_cost
    fn_loss_fraction = FN_LOSS_FRACTION if fn_loss_fraction is None else fn_loss_fraction
    fn_fixed_cost = FN_FIXED_COST if fn_fixed_cost is None else fn_fixed_cost

    y_true = np.asarray(y_true)
    amounts = np.nan_to_num(np.asarray(amounts, dtype=float))
    y_pred = (np.asarray(y_scores, dtype=float) >= threshold).astype(int)

    fn_mask = (y_true == 1) & (y_pred == 0)          # fraud we let through
    fp_mask = (y_true == 0) & (y_pred == 1)          # customers we wrongly stopped

    fn_cost = float(amounts[fn_mask].sum() * fn_loss_fraction
                    + fn_mask.sum() * fn_fixed_cost)
    fp_cost = float(fp_mask.sum() * fp_unit_cost)
    total = fn_cost + fp_cost
    n = len(y_true)
    return {
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_cost": total,
        "cost_per_million": float(total / n * 1_000_000) if n else 0.0,
        "n_missed_fraud": int(fn_mask.sum()),
        "n_false_declines": int(fp_mask.sum()),
    }


def best_threshold(y_true, y_scores, amounts, fp_unit_cost: float = None,
                   grid=None, **cost_kwargs):
    """Sweep thresholds, return the one minimising total cost.

    This is what turns a probability model into a DECISION. Choose it on
    VALIDATION data and merely apply it to test: choosing it on test optimises
    the cut point against the very data you are reporting, which is
    optimistically biased.

    Returns (threshold, total_cost, curve) where curve is [(t, cost), ...].
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    curve = [(float(t),
              cost(y_true, y_scores, amounts, t, fp_unit_cost, **cost_kwargs)["total_cost"])
             for t in grid]
    t_best, c_best = min(curve, key=lambda x: x[1])
    return t_best, c_best, curve


# ── convenience wrappers ─────────────────────────────────────────────────────
def evaluate(y_true, y_scores, amounts, threshold: float,
             fp_unit_cost: float = None, label: str = "", **cost_kwargs) -> dict:
    """Full picture at one threshold: ranking + threshold + cost, merged."""
    out = {**ranking_metrics(y_true, y_scores),
           **threshold_metrics(y_true, y_scores, threshold),
           **cost(y_true, y_scores, amounts, threshold, fp_unit_cost, **cost_kwargs)}
    if label:
        out["config"] = label
        logger.info(f"[{label}] AUROC={out['AUROC']:.4f} PR_AUC={out['PR_AUC']:.4f} "
                    f"recall={out['recall']:.3f} precision={out['precision']:.3f} "
                    f"cost/M=${out['cost_per_million']:,.0f}")
    return out


def compare(configs: dict, y_true, amounts, fp_unit_cost: float = None,
            **cost_kwargs) -> pd.DataFrame:
    """Score several configurations side by side on identical data.

    configs: {"rules-only": (scores, threshold), "model": (scores, threshold), ...}

    This is the adoption argument: the same test set, the same cost assumptions,
    one row per configuration.
    """
    rows = [evaluate(y_true, s, amounts, t, fp_unit_cost, label=name, **cost_kwargs)
            for name, (s, t) in configs.items()]
    df = pd.DataFrame(rows).set_index("config")
    cols = ["AUROC", "PR_AUC", "recall", "precision", "false_positive_rate",
            "n_missed_fraud", "n_false_declines", "cost_per_million"]
    return df[[c for c in cols if c in df.columns]]


def segment_metrics(y_true, y_scores, amounts, threshold: float, segments,
                    fp_unit_cost: float = None, min_n: int = 30,
                    **cost_kwargs) -> pd.DataFrame:
    """Break performance down by a categorical (paymentProvider, fraud_type, corridor).

    Worth doing routinely. A blended number hides heterogeneity: the model may be
    excellent on card-funded fraud and useless on scam-induced transfers, or strong
    for one provider and weak for another. If label COVERAGE differs by segment,
    an aggregate metric can also flatter the segment that simply has better labels.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    seg = pd.Series(segments).reset_index(drop=True).fillna("(missing)")

    rows = []
    for value, idx in seg.groupby(seg).groups.items():
        m = np.zeros(len(seg), dtype=bool)
        m[np.asarray(idx)] = True
        if m.sum() < min_n:
            continue
        r = {"segment": value, "n": int(m.sum()),
             "fraud_rate": float(y_true[m].mean())}
        r.update(ranking_metrics(y_true[m], y_scores[m]))
        r.update(threshold_metrics(y_true[m], y_scores[m], threshold))
        r.update(cost(y_true[m], y_scores[m], amounts[m], threshold,
                      fp_unit_cost, **cost_kwargs))
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("segment").sort_values("n", ascending=False)
    return df[["n", "fraud_rate", "AUROC", "PR_AUC", "recall", "precision",
               "cost_per_million"]]


def cost_grid(y_true, y_scores, amounts, fp_costs=None, grid=None,
              **cost_kwargs) -> dict:
    """Precompute cost-minimising threshold and cost across a range of
    fp_unit_cost assumptions.

    Feeds the demo's cost slider: recomputing over the full test set on every
    slider move is slow, so the curve is computed once and looked up.
    """
    fp_costs = fp_costs or [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    out = {}
    for fpc in fp_costs:
        t, c, _ = best_threshold(y_true, y_scores, amounts, fpc, grid, **cost_kwargs)
        tm = threshold_metrics(y_true, y_scores, t)
        cm = cost(y_true, y_scores, amounts, t, fpc, **cost_kwargs)
        out[str(fpc)] = {"threshold": t, "recall": tm["recall"],
                         "precision": tm["precision"],
                         "cost_per_million": cm["cost_per_million"]}
    return out