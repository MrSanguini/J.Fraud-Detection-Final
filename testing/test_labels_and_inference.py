"""
test_labels_and_inference.py — label derivation and the scoring seam.

Label logic is tested with hand-built flag documents rather than generated data,
so each rule is isolated and the intent is readable from the test itself.
"""

import json

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL DERIVATION
# ══════════════════════════════════════════════════════════════════════════════
def _flags(rows):
    """Build a flag table. Flag documents are APPEND-ONLY: a flag and a later
    unflag are two rows, not one mutated row."""
    return pd.DataFrame(rows)


def _transfers(ids):
    return pd.DataFrame({"_id": ids})


def test_single_flag_is_confirmed_fraud():
    import label_builder
    out = label_builder.build_labels(
        _flags([{"transfer": "t1", "action": "flag", "date": "2026-01-01T00:00:00Z"}]),
        _transfers(["t1"]))
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "label_source"] == label_builder.SOURCE_UPHELD


def test_latest_event_wins_flag_then_unflag():
    """A false positive: flagged, then cleared on review. The LATEST event is the
    verdict, so this is a confirmed LEGITIMATE transaction -- a clean negative."""
    import label_builder
    out = label_builder.build_labels(
        _flags([
            {"transfer": "t1", "action": "flag", "date": "2026-01-01T00:00:00Z"},
            {"transfer": "t1", "action": "unflag", "date": "2026-01-02T00:00:00Z"},
        ]),
        _transfers(["t1"]))
    assert out.loc[0, "label"] == 0
    assert out.loc[0, "label_source"] == label_builder.SOURCE_CLEARED


def test_latest_event_wins_re_flagged():
    """flag -> unflag -> re-flag must resolve to FRAUD. Reading the first or an
    arbitrary document instead of the latest would silently mislabel these."""
    import label_builder
    out = label_builder.build_labels(
        _flags([
            {"transfer": "t1", "action": "flag", "date": "2026-01-01T00:00:00Z"},
            {"transfer": "t1", "action": "unflag", "date": "2026-01-02T00:00:00Z"},
            {"transfer": "t1", "action": "flag", "date": "2026-01-03T00:00:00Z"},
        ]),
        _transfers(["t1"]))
    assert out.loc[0, "label"] == 1


def test_out_of_order_documents_still_resolve():
    """Export order is not chronological order. Sorting by date is what makes
    the append-only history correct."""
    import label_builder
    out = label_builder.build_labels(
        _flags([
            {"transfer": "t1", "action": "unflag", "date": "2026-01-03T00:00:00Z"},
            {"transfer": "t1", "action": "flag", "date": "2026-01-01T00:00:00Z"},
        ]),
        _transfers(["t1"]))
    assert out.loc[0, "label"] == 0          # the unflag is later


def test_unflagged_transfers_are_assumed_legit_and_marked():
    """THE BLIND SPOT. Never-flagged transfers are assumed legitimate, but the
    bucket contains the fraud the rules never caught. It must stay distinguishable
    from confirmed-legit so it can be weighted or excluded later."""
    import label_builder
    out = label_builder.build_labels(_flags([]), _transfers(["t1", "t2"]))
    assert (out["label"] == 0).all()
    assert (out["label_source"] == label_builder.SOURCE_ASSUMED).all()


def test_chargeback_overrides_a_human_clearance():
    """The most valuable training example available: review cleared it, the card
    provider later charged it back. That is a false negative OF THE REVIEW
    PROCESS, and external confirmation outranks internal review."""
    import label_builder
    out = label_builder.build_labels(
        _flags([
            {"transfer": "t1", "action": "flag", "date": "2026-01-01T00:00:00Z"},
            {"transfer": "t1", "action": "unflag", "date": "2026-01-02T00:00:00Z"},
        ]),
        _transfers(["t1"]),
        chargebacks=pd.DataFrame([{"_id": "t1", "confirmed_date": "2026-02-01T00:00:00Z"}]))
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "label_source"] == label_builder.SOURCE_CHARGEBACK


def test_chargeback_promotes_an_assumed_legit_transfer():
    """Fraud the rules never saw at all -- the case that makes the model a
    rules-IMPROVER rather than a rules-imitator."""
    import label_builder
    out = label_builder.build_labels(
        _flags([]), _transfers(["t1"]),
        chargebacks=pd.DataFrame([{"_id": "t1", "confirmed_date": "2026-02-01T00:00:00Z"}]))
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "label_source"] == label_builder.SOURCE_CHARGEBACK


def test_unrecognised_action_raises():
    """Fail loudly. Silently dropping unknown actions would corrupt labels while
    looking like a clean run."""
    import label_builder
    with pytest.raises(ValueError, match="[Uu]nexpected action"):
        label_builder.build_labels(
            _flags([{"transfer": "t1", "action": "quarantine",
                     "date": "2026-01-01T00:00:00Z"}]),
            _transfers(["t1"]))


def test_null_transfer_reference_raises():
    """The transfer-flag collection must reference a transfer on every row.
    A null means user/recipient flags got mixed in."""
    import label_builder
    with pytest.raises(ValueError, match="null 'transfer'"):
        label_builder.build_labels(
            _flags([{"transfer": None, "action": "flag",
                     "date": "2026-01-01T00:00:00Z"}]),
            _transfers(["t1"]))


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE SEAM
# ══════════════════════════════════════════════════════════════════════════════
def test_result_exposes_no_decision():
    """Decisioning lives in a separate system. This service returns risk, not a
    verdict; a `decision` field here would blur that boundary."""
    import inference
    d = inference.ScoreResult(probability=0.73).to_dict()
    assert "decision" not in d
    assert "rules_fired" not in d
    assert set(d) == {"probability", "risk_band", "risk_factors",
                      "behavioural_factors", "latency_ms"}


@pytest.mark.parametrize("prob,band", [
    (0.00, "Low"), (0.14, "Low"), (0.15, "Elevated"), (0.49, "Elevated"),
    (0.50, "High"), (0.84, "High"), (0.85, "Critical"), (1.00, "Critical"),
])
def test_risk_band_boundaries(prob, band):
    import inference
    assert inference.ScoreResult(probability=prob).risk_band == band


def test_feature_labels_are_human_readable():
    import inference
    assert inference.label_for("send_amount") == "Transfer amount"
    assert inference.label_for("ipDetails__security__vpn") == "VPN detected"
    assert inference.is_interpretable("send_amount")


def test_frequency_encoded_columns_stay_readable():
    """A _FE column of a readable field must not fall through to 'opaque'.
    Rarity IS the signal, and reviewers need it in plain language."""
    import inference
    assert inference.is_interpretable("recipient_bank_name_FE")
    assert "how common" in inference.label_for("recipient_bank_name_FE")


def test_opaque_features_are_flagged_as_behavioural():
    import inference
    assert not inference.is_interpretable("V258")
    assert inference.label_for("V258") == "Behavioural signal V258"


def test_to_dict_is_json_serialisable():
    """numpy scalars are not JSON-native; an API returning them would 500."""
    import inference
    r = inference.ScoreResult(
        probability=np.float64(0.42),
        risk_factors=[{"label": "Account age", "value": np.int64(37),
                       "impact": np.float32(0.8)}])
    json.dumps(r.to_dict(), default=float)      # must not raise


def test_explain_describes_risk_not_a_decision():
    import inference
    r = inference.ScoreResult(probability=0.87, risk_factors=[
        {"label": "Transfer amount", "value": 3200, "hidden_signal_weight": 0.0}])
    text = inference.explain(r)
    assert "Critical risk" in text
    assert "BLOCK" not in text.upper()


def test_band_distribution_separates_risk():
    """Higher bands must hold a higher fraud rate, or the score is not
    separating risk at all."""
    import inference
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    probs = np.clip(rng.beta(2, 6, 2000) + y * 0.5, 0, 1)
    df = inference.band_distribution(probs, y)
    assert df["count"].sum() == 2000
    high = df.loc["Critical", "fraud_rate_pct"] if df.loc["Critical", "count"] else 100
    assert high > df.loc["Low", "fraud_rate_pct"]
