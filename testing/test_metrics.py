"""
test_metrics.py — the cost model and evaluation harness.

These are pure unit tests: no data generation, no pipeline. They run in
milliseconds and catch arithmetic regressions in the numbers that drive every
threshold decision and every figure quoted to management.
"""

import numpy as np
import pytest

import metrics


# ── the money model ──────────────────────────────────────────────────────────
def test_cost_is_asymmetric(tiny_case):
    """Missed fraud costs its AMOUNT; a wrong decline costs a FLAT fee.

    This asymmetry is the whole point of the cost model: a $5 missed fraud and a
    $5,000 missed fraud are not the same mistake."""
    c = metrics.cost(tiny_case["y"], tiny_case["scores"], tiny_case["amounts"],
                     tiny_case["threshold"], fp_unit_cost=25.0)
    assert c["fn_cost"] == 500.0        # the one missed fraud, at its amount
    assert c["fp_cost"] == 25.0         # one wrong decline, flat
    assert c["total_cost"] == 525.0
    assert c["n_missed_fraud"] == 1
    assert c["n_false_declines"] == 1


def test_cost_per_million_scales(tiny_case):
    c = metrics.cost(tiny_case["y"], tiny_case["scores"], tiny_case["amounts"],
                     tiny_case["threshold"], fp_unit_cost=25.0)
    assert c["cost_per_million"] == pytest.approx(525.0 / 4 * 1_000_000)


def test_fn_loss_fraction(tiny_case):
    """If the business only eats a fraction of a fraudulent transfer, say so.

    Full principal is the remittance default (funds are disbursed and rarely
    recovered), but liability can sit elsewhere."""
    c = metrics.cost(tiny_case["y"], tiny_case["scores"], tiny_case["amounts"],
                     tiny_case["threshold"], fp_unit_cost=25.0,
                     fn_loss_fraction=0.02)
    assert c["fn_cost"] == pytest.approx(10.0)      # 2% of $500


def test_fn_fixed_cost(tiny_case):
    """Per-fraud overhead independent of amount: chargeback fees, investigation."""
    c = metrics.cost(tiny_case["y"], tiny_case["scores"], tiny_case["amounts"],
                     tiny_case["threshold"], fp_unit_cost=25.0, fn_fixed_cost=15.0)
    assert c["fn_cost"] == pytest.approx(515.0)     # 500 + 15


def test_cost_handles_nan_amounts():
    """A NaN amount must not poison the total."""
    c = metrics.cost([1, 0], [0.1, 0.9], [np.nan, 100], 0.5, fp_unit_cost=25.0)
    assert np.isfinite(c["total_cost"])


# ── threshold metrics ────────────────────────────────────────────────────────
def test_confusion_matrix(tiny_case):
    m = metrics.threshold_metrics(tiny_case["y"], tiny_case["scores"],
                                  tiny_case["threshold"])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)


def test_false_positive_rate_is_over_legit_only(tiny_case):
    """FPR must be fp/(fp+tn) -- the share of LEGITIMATE traffic wrongly stopped.

    This is the number that compares against the rules engine's ~80%, so getting
    the denominator wrong would misstate the headline claim."""
    m = metrics.threshold_metrics(tiny_case["y"], tiny_case["scores"],
                                  tiny_case["threshold"])
    assert m["false_positive_rate"] == pytest.approx(1 / 2)


def test_binary_decisions_work():
    """Rules-only configurations pass 0/1 decisions with threshold 0.5."""
    m = metrics.threshold_metrics([1, 0, 1, 0], [1, 1, 0, 0], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)


# ── threshold selection ──────────────────────────────────────────────────────
@pytest.fixture
def scored_population():
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.02).astype(int)
    scores = np.clip(rng.beta(2, 5, n) + y * 0.35, 0, 1)
    amounts = rng.lognormal(4.3, 1.1, n)
    return y, scores, amounts


def test_best_threshold_returns_the_minimum(scored_population):
    y, s, a = scored_population
    t, best, curve = metrics.best_threshold(y, s, a, fp_unit_cost=25.0)
    assert best == pytest.approx(min(c for _, c in curve))
    assert any(abs(t - tt) < 1e-9 for tt, _ in curve)


def test_threshold_rises_with_false_positive_cost(scored_population):
    """The more a wrong decline costs, the more selective the model should be.

    Directional check rather than an exact value: it catches a sign error in the
    cost function, which would otherwise look plausible."""
    y, s, a = scored_population
    t_cheap, _, _ = metrics.best_threshold(y, s, a, fp_unit_cost=5.0)
    t_dear, _, _ = metrics.best_threshold(y, s, a, fp_unit_cost=200.0)
    assert t_dear > t_cheap


# ── ranking metrics ──────────────────────────────────────────────────────────
def test_pr_auc_baseline_is_the_fraud_rate(scored_population):
    """PR-AUC's random baseline IS the positive rate. Reporting it stops anyone
    reading 0.45 as 'bad' when random would score 0.02."""
    y, s, _ = scored_population
    r = metrics.ranking_metrics(y, s)
    assert r["pr_auc_baseline"] == pytest.approx(y.mean())
    assert r["PR_AUC"] > r["pr_auc_baseline"]


def test_ranking_metrics_survive_single_class():
    """Must not raise on a degenerate slice -- segment reports hit this."""
    r = metrics.ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(r["AUROC"])


def test_perfect_and_inverted_scores():
    y = [0, 0, 1, 1]
    assert metrics.ranking_metrics(y, [0.1, 0.2, 0.8, 0.9])["AUROC"] == pytest.approx(1.0)
    assert metrics.ranking_metrics(y, [0.9, 0.8, 0.2, 0.1])["AUROC"] == pytest.approx(0.0)


# ── reporting helpers ────────────────────────────────────────────────────────
def test_compare_returns_one_row_per_config(scored_population):
    y, s, a = scored_population
    rules = (s > 0.6).astype(int)
    df = metrics.compare({"rules": (rules, 0.5), "model": (s, 0.3)}, y, a,
                         fp_unit_cost=25.0)
    assert list(df.index) == ["rules", "model"]
    assert "cost_per_million" in df.columns


def test_segment_metrics_skips_tiny_segments(scored_population):
    """Segments below min_n are dropped: metrics on 3 rows are noise, and
    publishing them invites false conclusions."""
    y, s, a = scored_population
    seg = np.array(["big"] * (len(y) - 5) + ["tiny"] * 5)
    df = metrics.segment_metrics(y, s, a, 0.3, seg, min_n=30)
    assert "tiny" not in df.index
    assert "big" in df.index


def test_cost_grid_covers_requested_costs(scored_population):
    y, s, a = scored_population
    g = metrics.cost_grid(y, s, a, fp_costs=[10, 50])
    assert set(g) == {"10", "50"}
    assert all("threshold" in v and "cost_per_million" in v for v in g.values())
