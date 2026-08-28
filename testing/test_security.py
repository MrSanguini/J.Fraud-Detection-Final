"""
test_security.py — the tests that matter most.

Every failure mode here is SILENT. A leaked SSN does not raise an exception; a
model trained on a leaked label just posts suspiciously good numbers; a salt that
loads differently produces zero chargeback matches and looks like "we have no
labels". These assertions are how those become loud.

Marked `security`. Do not skip them.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.security


# ── pseudonymisation ─────────────────────────────────────────────────────────
def test_hash_is_deterministic_within_process():
    """Joins depend on the same identity hashing to the same token."""
    from common import hash_id
    salt = b"fixed-salt"
    assert hash_id("user-123", salt) == hash_id("user-123", salt)


def test_hash_is_deterministic_across_processes():
    """THE SILENT KILLER.

    The extractor and the chargeback parser run separately. If the salt resolves
    differently between them, every hash differs, every join returns zero rows,
    and the symptom is 'no chargebacks matched' -- which reads like missing data,
    not a bug. This runs the hash in a fresh interpreter to prove stability."""
    from common import hash_id
    salt_str = "cross-process-salt"
    here = hash_id("user-123", salt_str.encode())

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from common import hash_id;"
        "print(hash_id('user-123', b'%s'))" % (ROOT / "pipeline", salt_str)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == here


def test_different_salts_give_different_hashes():
    from common import hash_id
    assert hash_id("user-123", b"salt-a") != hash_id("user-123", b"salt-b")


def test_hash_normalises_case_and_whitespace():
    """Partner sheets and Mongo exports format ids inconsistently. Normalising
    before hashing is what lets a FOLIO match its transfer."""
    from common import hash_id
    salt = b"s"
    assert hash_id("  ABC123  ", salt) == hash_id("abc123", salt)


def test_hash_of_none_is_none():
    from common import hash_id
    assert hash_id(None, b"s") is None
    assert hash_id("", b"s") is None


def test_hash_is_not_trivially_reversible():
    """Salted SHA-256, not a plain digest. Pseudonymous, not anonymous -- but an
    unsalted hash of a low-entropy id would be brute-forceable in seconds."""
    from common import hash_id
    plain = hashlib.sha256(b"user-123").hexdigest()[:24]
    assert hash_id("user-123", b"some-salt") != plain


# ── PII exclusion ────────────────────────────────────────────────────────────
FORBIDDEN = ["ssn", "password", "passwordreset", "token", "secret",
             "first_name", "last_name", "full_name", "sender_name",
             "recipient_name", "phone_number", "account_number",
             "last_account_num", "state_id", "plaid", "latitude", "longitude"]

ALLOWED_EXACT = {
    "sender_email_domain", "recipient_email_domain", "ip_country_mismatch",
    "ipDetails__security__vpn", "ipDetails__security__proxy",
    "ipDetails__security__tor", "ipDetails__security__relay",
    "ipDetails__location__country_code", "ipDetails__location__city",
    "ipDetails__network__autonomous_system_number", "billing_country_match",
}


@pytest.mark.slow
def test_no_pii_columns_in_training_set(training_set):
    """The allowlist should make PII structurally impossible. Verify it."""
    offenders = []
    for c in training_set.columns:
        if c in ALLOWED_EXACT:
            continue
        low = c.lower()
        for bad in FORBIDDEN:
            if bad in low:
                offenders.append(f"{c} (matched {bad!r})")
    assert not offenders, f"PII reached the training set: {offenders}"


@pytest.mark.slow
def test_no_identity_columns_in_training_set(training_set):
    """Hashed ids do their job during aggregation, then must leave. A model that
    can see an account id can memorise accounts instead of learning behaviour."""
    identities = {"_id", "user_id", "recipient_id", "_t",
                  "partner_reference", "partner_txn_id"}
    present = identities & set(training_set.columns)
    assert not present, f"identity columns survived: {present}"


@pytest.mark.slow
def test_no_staging_columns_survive(training_set):
    """_raw_ columns are staging for derivations. Any that survive are carrying
    unprocessed source values -- including, potentially, sensitive ones."""
    raw = [c for c in training_set.columns if "_raw_" in c]
    assert not raw, f"unconsumed staging columns: {raw}"


@pytest.mark.slow
def test_pii_canary(synthetic_dir, tmp_path):
    """Inject a document containing a fake SSN and prove it does not come out.

    Asserting the ABSENCE of something is weak on its own -- a broken extractor
    that emitted nothing would also pass. This proves the allowlist actively
    excludes a field that IS present in the source."""
    from common import load_json
    import extractor

    users = load_json(synthetic_dir / "users.json")
    assert users, "no users generated"
    assert "ssn" in users[0], "canary precondition failed: source has no ssn field"
    canary = "999-CANARY-999"
    users[0]["ssn"] = canary

    df, _audit, _canon = extractor.extract_collection(users, "user")
    blob = df.to_csv(index=False)
    assert canary not in blob, "SSN canary reached the extracted table"
    assert not [c for c in df.columns if "ssn" in c.lower()]


# ── label leakage ────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_label_source_maps_perfectly_to_label(training_set):
    """label_source is PERFECT leakage: 'confirmed_fraud_chargeback' is always
    fraud. A model given it scores ~1.0 and is worthless. This documents WHY it
    is excluded, and fails loudly if that relationship ever changes."""
    xt = pd.crosstab(training_set["label_source"], training_set["label"])
    assert ((xt > 0).sum(axis=1) == 1).all(), \
        "label_source no longer maps 1:1 to label -- re-check the exclusion rationale"


@pytest.mark.slow
def test_metadata_absent_from_feature_matrix(prepared):
    """The training set legitimately carries metadata. The guarantee is that it
    never reaches the MODEL."""
    import model_prep
    cols = set(prepared["X_train"].columns)
    leaked = [m for m in model_prep.METADATA_COLS if m in cols]
    assert not leaked, f"metadata reached the feature matrix: {leaked}"


@pytest.mark.slow
def test_amounts_returned_separately_from_features(prepared):
    """amount_paid drives the cost model but is unknown at scoring time. It must
    travel beside the features, never inside them."""
    assert "amounts_test" in prepared
    assert len(prepared["amounts_test"]) == len(prepared["X_test"])
    assert "amount_paid" not in prepared["X_test"].columns


# ── point-in-time integrity ──────────────────────────────────────────────────
@pytest.mark.slow
def test_velocity_excludes_the_current_row():
    """closed="left" is THE leakage guard in the pipeline.

    Without it a transaction is counted in its own 'transfers in the last hour',
    so the model sees a trace of the row it is predicting. A user's FIRST
    transaction must therefore have zero prior history."""
    import assembler

    base = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2"],
        "date": [base, base + pd.Timedelta(minutes=10),
                 base + pd.Timedelta(minutes=20), base],
        "send_amount": [100.0, 200.0, 300.0, 50.0],
    })
    out = assembler.derive_time_features(df)
    out = assembler.add_velocity_features(out, key="user_id", windows=(24,))
    out = out.sort_values(["user_id", "_t"])

    firsts = out.groupby("user_id").head(1)
    assert (firsts["user_id_txn_24h"] == 0).all(), "first transaction has prior history"
    assert (firsts["user_id_amt_24h"] == 0).all()

    u1 = out[out["user_id"] == "u1"].reset_index(drop=True)
    assert u1.loc[1, "user_id_txn_24h"] == 1      # sees only the first
    assert u1.loc[2, "user_id_amt_24h"] == 300.0  # 100 + 200, NOT its own 300


@pytest.mark.slow
def test_recipient_history_counts_prior_sends_only():
    import assembler
    base = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u1"],
        "recipient_id": ["r1", "r1", "r2"],
        "date": [base, base + pd.Timedelta(days=1), base + pd.Timedelta(days=2)],
        "send_amount": [10.0, 20.0, 30.0],
    })
    out = assembler.derive_time_features(df)
    out = assembler.add_recipient_history(out).reset_index(drop=True)
    assert out.loc[0, "is_first_send_to_recipient"] == 1
    assert out.loc[1, "is_first_send_to_recipient"] == 0   # same recipient again
    assert out.loc[2, "is_first_send_to_recipient"] == 1   # different recipient


@pytest.mark.slow
def test_temporal_split_does_not_shuffle(prepared, training_set):
    """Train must precede test in TIME. A random split scatters future
    transactions into training and inflates every metric."""
    import model_prep
    train, val, test = model_prep.temporal_split(training_set)
    t_train = pd.to_datetime(train["date"], utc=True, format="mixed").max()
    t_test = pd.to_datetime(test["date"], utc=True, format="mixed").min()
    assert t_train <= t_test, "test data predates the end of training"


@pytest.mark.slow
def test_row_count_reconciles(training_set, synthetic_dir):
    """Silent row loss from a bad join is invisible unless asserted."""
    from common import load_json
    n_source = len(load_json(synthetic_dir / "transfers.json"))
    assert len(training_set) == n_source, \
        f"expected {n_source} rows, got {len(training_set)} -- a join dropped rows"
