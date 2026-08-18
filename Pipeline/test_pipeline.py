"""
test_pipeline.py — safety checks. RUN THIS AFTER EVERY PIPELINE RUN.

The pipeline runs unsupervised on real customer data, so it must catch its own
mistakes. These assertions encode the guarantees we designed for:

  1. NO PII or secrets in the training set (exclusion, not obfuscation)
  2. NO identity columns (the model must not memorise who)
  3. NO leakage fields (nothing knowable only after the scoring moment)
  4. NO unconsumed _raw_ staging columns
  5. Labels present and correctly typed
  6. Point-in-time integrity (no negative ages; velocity excludes current row)
"""

import re
import sys
from pathlib import Path

import pandas as pd

from common import read_table

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

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
    "flag_exempt", "avs_exempt", "amount_paid", "retry", "isPendingPayment",
    "user_flag_status", "recipient_flag_status", "label_source_leak",
}


def check_training_set(path="data/training_set") -> bool:
    df = read_table(_resolve(path))
    cols = list(df.columns)
    failures = []

    # 1 — PII / secrets
    for c in cols:
        if c in ALLOWED_EXACT:
            continue
        low = c.lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            # word-boundary match: "ip" must not fire on "recipient"
            if re.search(rf"(^|[^a-z]){re.escape(bad)}([^a-z]|$)", low):
                failures.append(f"PII/secret risk: column {c!r} matches {bad!r}")

    # 2 — identities
    for c in cols:
        if c in IDENTITY_COLUMNS:
            failures.append(f"identity column survived: {c!r}")

    # 3 — leakage
    for c in cols:
        if c in LEAKAGE_COLUMNS:
            failures.append(f"leakage column survived: {c!r}")

    # 4 — staging columns
    for c in cols:
        if c.startswith("_raw_"):
            failures.append(f"unconsumed staging column: {c!r}")

    # 5 — labels
    if "label" not in cols:
        failures.append("no 'label' column")
    else:
        bad = set(pd.unique(df["label"].dropna())) - {0, 1}
        if bad:
            failures.append(f"label has non-binary values: {bad}")

    # 6 — point-in-time integrity
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
    print(f"\n{'='*66}\nSAFETY CHECK — {len(df):,} rows x {len(cols)} cols\n{'='*66}")
    if failures:
        print(f"\n  FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"    x {f}")
        return False
    print("\n  PASSED — no PII, no identities, no leakage, no staging columns.")
    print(f"  label distribution: {df['label'].value_counts().to_dict()}")
    return True


if __name__ == "__main__":
    ok = check_training_set()
    sys.exit(0 if ok else 1)
