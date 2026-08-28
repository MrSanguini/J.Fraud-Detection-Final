"""
test_pipeline.py: safety checks. RUN THIS AFTER EVERY PIPELINE RUN.

The pipeline runs unsupervised on real customer data, so it must catch its own
mistakes. These assertions encode the guarantees we designed for:

  1. NO PII or secrets in the training set (exclusion, not obfuscation)
  2. NO identity columns (the model must not memorise who)
  3. NO leakage fields (nothing knowable only after the scoring moment)
  4. NO unconsumed _raw_ staging columns
  5. Labels present and correctly typed
  6. Point-in-time integrity (no negative ages; velocity excludes current row)
"""

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import config
from common import read_table, setup_logging

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = _P(p)
    return p if p.is_absolute() else ROOT / p


# substrings that must NEVER appear in a training-set column name
FORBIDDEN_SUBSTRINGS = [
    "ssn", "password", "passwordreset", "pin", "token", "secret",
    "first_name", "last_name", "full_name", "sender_name", "recipient_name",
    "phone_number", "account_number", "last_account_num", "address",
    "email",            # domains are allowed via an explicit exception below
    "ip", "latitude", "longitude", "paymentlink", "receipturl", "callback",
    "state_id", "plaid",
]
# explicit exceptions: derived, non-identifying
ALLOWED_EXACT = {
    "sender_email_domain", "recipient_email_domain",
    "ip_country_mismatch",
    "ipDetails__security__vpn", "ipDetails__security__proxy",
    "ipDetails__security__tor", "ipDetails__security__relay",
    "ipDetails__location__country_code", "ipDetails__location__city",
    "ipDetails__network__autonomous_system_number",
    "billing_country_match",
    "user_addressChangeCounter", "user_pinFailedAttemptsCount",
    "user_isPinLocked",
}

IDENTITY_COLUMNS = {"_id", "user_id", "recipient_id", "_t",
                    "partner_reference", "partner_txn_id",
                    "user_user_id", "recipient_user_id"}

LEAKAGE_COLUMNS = {
    "status", "payment_status", "payment_partner_status", "payout_partner_status",
    "flag_status", "date_flagged", "isMonitored", "date_monitored",
    "flag_exempt", "avs_exempt", "retry", "isPendingPayment",
    "user_flag_status", "recipient_flag_status", "label_source_leak",
    # amount_paid is NOT here: assembler.assemble() deliberately retains it
    # (alongside date, label_source) as training_set-level metadata for the
    # cost model. It must never reach the FEATURE MATRIX though -- that's
    # what check_feature_matrix() below actually proves, on the real output
    # of model_prep.prepare(), rather than trusting this column list.
}

# Independent of model_prep.METADATA_COLS on purpose: check_feature_matrix()
# below must catch it if that upstream list is ever accidentally shrunk, not
# just verify the feature matrix agrees with whatever it currently says.
EXPECTED_METADATA_EXCLUSIONS = ["label", "label_source", "date", "amount_paid"]


def check_training_set(path=None) -> bool:
    setup_logging()
    t0 = time.monotonic()
    df = read_table(_resolve(path or config.TRAINING_SET_PATH))
    cols = list(df.columns)
    failures = []

    # 1. PII / secrets
    for c in cols:
        if c in ALLOWED_EXACT:
            continue
        low = c.lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            # word-boundary match: "ip" must not fire on "recipient"
            if re.search(rf"(^|[^a-z]){re.escape(bad)}([^a-z]|$)", low):
                failures.append(f"PII/secret risk: column {c!r} matches {bad!r}")

    # 2. identities
    for c in cols:
        if c in IDENTITY_COLUMNS:
            failures.append(f"identity column survived: {c!r}")

    # 3. leakage
    for c in cols:
        if c in LEAKAGE_COLUMNS:
            failures.append(f"leakage column survived: {c!r}")

    # 4. staging columns
    for c in cols:
        if c.startswith("_raw_"):
            failures.append(f"unconsumed staging column: {c!r}")

    # 5. labels
    if "label" not in cols:
        failures.append("no 'label' column")
    else:
        bad = set(pd.unique(df["label"].dropna())) - {0, 1}
        if bad:
            failures.append(f"label has non-binary values: {bad}")

    # 6. point-in-time integrity
    for age in ["user_age_days", "recipient_age_days"]:
        if age in cols:
            neg = (pd.to_numeric(df[age], errors="coerce") < 0).sum()
            if neg:
                failures.append(f"{neg} rows have negative {age} "
                                f"(record created after the transaction)")
    if "prior_sends_to_recipient" in cols:
        if (pd.to_numeric(df["prior_sends_to_recipient"], errors="coerce") < 0).any():
            failures.append("negative prior_sends_to_recipient")

    # ── report ─────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    logger.info(f"SAFETY CHECK: {len(df):,} rows x {len(cols)} cols ({elapsed:.2f}s)")
    if failures:
        logger.error(f"FAILED ({len(failures)} issue(s)):")
        for f in failures:
            logger.error(f"    x {f}")
        return False
    logger.info("PASSED: no PII, no identities, no leakage, no staging columns.")
    logger.info(f"label distribution: {df['label'].value_counts().to_dict()}")
    return True


def check_feature_matrix(training_set_path=None) -> bool:
    """Build the ACTUAL feature matrix via model_prep.prepare() and assert
    EXPECTED_METADATA_EXCLUSIONS (label, label_source, date, amount_paid)
    are absent from it.

    training_set legitimately carries those as metadata -- see
    assembler.assemble()'s comment on why amount_paid/date/label_source
    survive there -- so the leakage guarantee has to be proven on the
    feature matrix a model would actually train on, not on training_set
    itself (check_training_set() above can't catch this; it never runs
    model_prep at all).

    Checked against EXPECTED_METADATA_EXCLUSIONS, a list hardcoded in THIS
    file, not model_prep.METADATA_COLS: model_prep.prepare() already
    asserts the latter internally, so re-checking the same list here would
    only catch a regression in the exclusion LOGIC, not one where
    METADATA_COLS itself gets accidentally shrunk. Checking a fixed,
    independent expectation catches both."""
    import model_prep

    try:
        out = model_prep.prepare(training_set_path, artifacts_dir=False)
    except Exception:
        logger.exception("[test:feature_matrix] FAILED, model_prep.prepare() raised")
        return False

    leaked = [m for m in EXPECTED_METADATA_EXCLUSIONS if m in out["X_train"].columns]
    ok = not leaked
    if ok:
        logger.info(f"[test:feature_matrix] PASSED, {len(EXPECTED_METADATA_EXCLUSIONS)} "
                    f"metadata columns absent from the {out['X_train'].shape[1]}-column "
                    f"feature matrix")
    else:
        logger.error(f"[test:feature_matrix] FAILED, metadata leaked into features: {leaked}")
    return ok


def test_hash_consistency() -> bool:
    """Hash the same value in two SEPARATE Python processes and assert the
    result is identical.

    Catches: the salt loading differently between call sites (e.g.
    extractor.py vs chargeback_parser.py) -- which raises nowhere. Every
    hashed join key would just come out different, and chargeback matching
    would silently fall to 0% with no error, only the "[WARN] low match
    rate" hint that's easy to miss on an unattended run.

    Two real subprocesses, not two in-process calls: an in-process call
    would still pass even if get_salt() were resolved once per process and
    cached in a way that only breaks across process boundaries."""
    pipeline_dir = _P(__file__).resolve().parent
    script = (
        f"import sys; sys.path.insert(0, {str(pipeline_dir)!r}); "
        "from common import get_salt, hash_id; "
        "print(hash_id('L1.9-hash-consistency-canary', get_salt()))"
    )
    outputs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        if r.returncode != 0:
            logger.error(f"[test:hash_consistency] FAILED, subprocess errored:\n{r.stderr.strip()}")
            return False
        outputs.append(r.stdout.strip())

    ok = bool(outputs[0]) and outputs[0] == outputs[1]
    if ok:
        logger.info(f"[test:hash_consistency] PASSED, stable across processes ({outputs[0][:12]}...)")
    else:
        logger.error(f"[test:hash_consistency] FAILED, got {outputs} from two processes")
    return ok


def test_leakage_canary() -> bool:
    """Inject a fake user document carrying a real ssn/password, run it
    through the ACTUAL extractor (not a check on schema_contract.py's
    declared actions), and assert the sentinel values appear nowhere in
    the output: not as a column name, not as a cell value anywhere else.

    Catches: an allowlist regression. This proves exclusion works at
    runtime rather than assuming the contract's declared DROP is honoured
    -- flip "ssn"'s action to KEEP in schema_contract.py and this fails."""
    import extractor

    ssn_sentinel = "L19-CANARY-SSN-078-05-1120"
    password_sentinel = "L19-CANARY-PASSWORD-hunter2"

    canary_doc = {
        "_id": {"$oid": "0" * 24},
        "dateCreated": {"$date": "2020-01-01T00:00:00Z"},
        "ssn": ssn_sentinel,
        "password": password_sentinel,
        "email": "canary@example.com",
        "first_name": "Canary",
    }

    df, _audit, _canon = extractor.extract_collection([canary_doc], "user")

    leaked_columns = [c for c in df.columns if c.lower() in ("ssn", "password")]
    as_text = df.astype(str)
    cell_leak = bool(as_text.apply(
        lambda col: col.str.contains(re.escape(ssn_sentinel), regex=True)
                   | col.str.contains(re.escape(password_sentinel), regex=True)
    ).to_numpy().any())

    ok = not leaked_columns and not cell_leak
    if ok:
        logger.info("[test:leakage_canary] PASSED, ssn/password absent from extractor output")
    else:
        logger.error(f"[test:leakage_canary] FAILED, leaked_columns={leaked_columns}, "
                     f"sentinel_value_leaked={cell_leak}")
    return ok


def test_row_count_reconciliation(extracted_dir=None, training_set_path=None) -> bool:
    """assert len(training_set) == len(transfers).

    Catches: silent row loss (or gain) from a bad join. Every merge in
    assembler.assemble() is a LEFT join FROM the transfer table, so the row
    count must be preserved exactly -- fewer rows means something dropped
    transfers silently; more rows means a join key fanned out on a
    duplicate on the right-hand side (user/recipient/labels). Either
    direction is a correctness bug, not a warning-log situation."""
    d = _resolve(extracted_dir or config.EXTRACTED_DIR)
    transfers = read_table(d / "transfer")
    training_set = read_table(_resolve(training_set_path or config.TRAINING_SET_PATH))

    n_tr, n_ts = len(transfers), len(training_set)
    ok = n_tr == n_ts
    if ok:
        logger.info(f"[test:row_count_reconciliation] PASSED, {n_ts:,} rows in both "
                    f"transfers and training_set")
    else:
        logger.error(f"[test:row_count_reconciliation] FAILED, transfers={n_tr:,} "
                     f"training_set={n_ts:,} (diff={n_ts - n_tr:+,})")
    return ok


def run_safety_tests(path=None, extracted_dir=None) -> bool:
    """Every pipeline safety assertion in one gate: the schema-level checks
    in check_training_set() plus the feature-matrix and process/security/join
    canaries above. This is what run_pipeline.py's Stage E calls."""
    setup_logging()
    results = {
        "training_set_checks":     check_training_set(path),
        "feature_matrix_checks":   check_feature_matrix(path),
        "hash_consistency":        test_hash_consistency(),
        "leakage_canary":          test_leakage_canary(),
        "row_count_reconciliation": test_row_count_reconciliation(extracted_dir, path),
    }
    n_passed = sum(results.values())
    logger.info(f"[safety tests] {n_passed}/{len(results)} passed: {results}")
    return n_passed == len(results)


if __name__ == "__main__":
    setup_logging()
    ok = run_safety_tests()
    sys.exit(0 if ok else 1)
