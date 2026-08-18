"""
assembler.py — Phase D: build the final training table.

Joins transfers + user + recipient + labels, derives features, and computes
behavioural aggregates.

THE CARDINAL RULE
-----------------
Every feature for a transaction at time T must be computable using ONLY
information available at time T.

Enforced by:
  * rolling windows use closed="left" — a transaction is never counted in its
    own velocity aggregate;
  * recipient history uses cumcount() — counts PRIOR sends only;
  * user/recipient documents are NOT snapshotted historically, so only
    IMMUTABLE/STATIC fields (creation dates) are trusted on old records.
    Tier-2 mutable fields are included conditionally and are NaN on old rows —
    the intended graceful degradation.

Identities are dropped at the end: they do their job during aggregation and
never reach the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import read_table, write_table

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = _P(p)
    return p if p.is_absolute() else ROOT / p


TIME = "_t"


# ── derivations from parked _raw_ values ─────────────────────────────────────
def derive_time_features(df: pd.DataFrame, time_col="date") -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[time_col], errors="coerce", utc=True, format="mixed")
    bad = t.isna().sum()
    if bad:
        print(f"[WARN] {bad} rows have an unparseable transaction date — they "
              f"cannot be ordered and will sort last.")
    out[TIME] = t
    out["hour"] = t.dt.hour.astype("Int16")
    out["weekday"] = t.dt.dayofweek.astype("Int16")
    return out


def add_account_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Ages are Tier-1 SAFE on all records: creation dates are immutable, so
    (transaction time - creation time) is correct even from a current snapshot.
    recipient_age_days is one of the strongest remittance fraud signals
    ('recipient created minutes before a large first-time transfer')."""
    out = df.copy()
    for src, name in [("user__raw_dateCreated", "user_age_days"),
                      ("recipient__raw_date_created", "recipient_age_days"),
                      ("user__raw_dob", "user_age_years")]:
        if src not in out.columns:
            continue
        created = pd.to_datetime(out[src], errors="coerce", utc=True, format="mixed")
        days = (out[TIME] - created).dt.total_seconds() / 86400
        out[name] = days / 365.25 if name.endswith("_years") else days
        neg = (out[name] < 0).sum()
        if neg:
            print(f"[WARN] {neg} rows have negative {name} — the referenced "
                  f"record was created AFTER the transaction. Check timestamps.")
    return out


def derive_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier-2 user derivations. NaN on historical rows by design; they populate
    automatically once transfer-time snapshots start arriving."""
    out = df.copy()
    lvl = pd.to_numeric(out.get("user__raw_threshold_level"), errors="coerce")
    rem = pd.to_numeric(out.get("user__raw_remainingBalance"), errors="coerce")
    if lvl is not None and rem is not None:
        total = lvl + rem                     # transacted + headroom = window limit
        out["limit_utilisation"] = np.where(total > 0, lvl / total, np.nan)

    if "user__raw_billing_address" in out.columns and "user_country" in out.columns:
        billing = out["user__raw_billing_address"].astype("string").str.strip().str.upper()
        acct = out["user_country"].astype("string").str.strip().str.upper()
        # element-wise: str.endswith cannot take a Series
        match = [
            (b.endswith(a) if isinstance(b, str) and isinstance(a, str) and a else np.nan)
            for b, a in zip(billing.tolist(), acct.tolist())
        ]
        out["billing_country_match"] = pd.Series(match, index=out.index, dtype="float64")
    return out


def derive_transfer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "_raw_previous_recipient" in out.columns:
        out["n_previous_recipients"] = out["_raw_previous_recipient"].apply(
            lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0)

    if "_raw_isEnabled3Ds" in out.columns:
        # null means the partner never reported it — that nullness is informative
        out["threeds_reported"] = out["_raw_isEnabled3Ds"].notna().astype(int)

    if "_raw_ipDetails__location__country" in out.columns and "sending_from" in out.columns:
        ipc = out["_raw_ipDetails__location__country"].astype("string").str.strip().str.lower()
        dec = out["sending_from"].astype("string").str.strip().str.lower()
        known = ipc.notna() & dec.notna()
        # prefix compare handles "United Kingdom of Great Britain..." vs "United Kingdom"
        same = ipc.fillna("").str[:6] == dec.fillna("").str[:6]
        out["ip_country_mismatch"] = np.where(known, ~same, np.nan).astype(float)

    # cents pattern: round amounts / repeated fractions are a known fraud tell
    if "send_amount" in out.columns:
        amt = pd.to_numeric(out["send_amount"], errors="coerce")
        out["cents"] = (amt - np.floor(amt)).astype(float)

    # strip ALL staging columns, including prefixed ones
    # (add_prefix turns "_raw_x" into "user__raw_x")
    return out.drop(columns=[c for c in out.columns if "_raw_" in c])


# ── point-in-time behavioural aggregates ─────────────────────────────────────
def add_velocity_features(df: pd.DataFrame, key="user_id",
                          windows=(1, 24, 168)) -> pd.DataFrame:
    """Rolling counts/sums over PRIOR transactions for the same key.

    closed="left" excludes the current row from its own aggregate. This single
    parameter is the main leakage guard in the pipeline."""
    out = df.sort_values(TIME).copy()
    if key not in out.columns or out[TIME].isna().all():
        return out

    out = out.reset_index(drop=True)
    for hours in windows:
        cnt_col, sum_col = f"{key}_txn_{hours}h", f"{key}_amt_{hours}h"
        counts = pd.Series(np.nan, index=out.index)
        sums = pd.Series(np.nan, index=out.index)
        for _, grp in out.groupby(key, sort=False):
            g = grp.dropna(subset=[TIME])
            if g.empty:
                continue
            ser = g.set_index(TIME)["send_amount"].astype(float)
            r = ser.rolling(f"{hours}h", closed="left")
            counts.loc[g.index] = r.count().to_numpy()
            sums.loc[g.index] = r.sum().to_numpy()
        out[cnt_col] = counts.fillna(0)
        out[sum_col] = sums.fillna(0.0)
    return out


def add_recipient_history(df: pd.DataFrame) -> pd.DataFrame:
    """First time this user has sent to this recipient? Cumulative in time
    order, excluding the current row."""
    out = df.sort_values(TIME).copy()
    if not {"user_id", "recipient_id"} <= set(out.columns):
        return out
    pair = out["user_id"].astype(str) + "|" + out["recipient_id"].astype(str)
    out["prior_sends_to_recipient"] = pair.groupby(pair).cumcount()
    out["is_first_send_to_recipient"] = (out["prior_sends_to_recipient"] == 0).astype(int)
    return out


# ── assembly ─────────────────────────────────────────────────────────────────
def assemble(extracted_dir="data/extracted") -> pd.DataFrame:
    d = _resolve(extracted_dir)
    tr = read_table(d / "transfer")
    lb = read_table(d / "labels")
    try:
        us = read_table(d / "user").add_prefix("user_").rename(columns={"user__id": "user_id"})
    except FileNotFoundError:
        us = None
    try:
        rc = read_table(d / "recipient").add_prefix("recipient_").rename(
            columns={"recipient__id": "recipient_id"})
        # avoid clobbering the transfer's own email-domain column
        rc = rc.rename(columns={
            "recipient_recipient_email_domain": "recipient_email_domain",
            # avoid collision with the transfer doc's own recipient_bank_name
            "recipient_bank_name": "recipient_doc_bank_name",
        })
    except FileNotFoundError:
        rc = None

    df = tr
    if us is not None:
        df = df.merge(us, on="user_id", how="left")
    if rc is not None:
        df = df.merge(rc, on="recipient_id", how="left")
    df = df.merge(lb[["_id", "label", "label_source"]], on="_id", how="left")

    ucol, rcol = "user__raw_dateCreated", "recipient__raw_date_created"
    um = df[ucol].notna().mean()*100 if ucol in df.columns else float("nan")
    rm = df[rcol].notna().mean()*100 if rcol in df.columns else float("nan")
    print(f"[assemble] {len(df):,} transfers | user match {um:.1f}% | "
          f"recipient match {rm:.1f}%")
    if um < 50 or rm < 50:
        print("   [WARN] low join match rate — check that user/recipient exports "
              "cover the same period as the transfers.")

    # ORDER MATTERS: time first (ages need _t), then age/user derivations
    # (they consume _raw_ columns), then transfer derivations (which strip _raw_).
    df = derive_time_features(df)
    df = add_account_ages(df)
    df = derive_user_features(df)
    df = derive_transfer_features(df)
    df = add_velocity_features(df, key="user_id")
    df = add_recipient_history(df)

    # identities and internals never reach the model
    drop = ["_id", "user_id", "recipient_id", TIME,
            "user_user_id", "recipient_user_id",
            "partner_reference", "partner_txn_id", "amount_paid"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df


def run(extracted_dir="data/extracted", out="data/training_set"):
    df = assemble(extracted_dir)
    out_path = write_table(df, _resolve(out))
    print(f"[write] {out_path} — {df.shape[0]:,} rows x {df.shape[1]} cols")
    if "label" in df:
        print(f"[label] fraud rate {df['label'].mean()*100:.3f}%")
    return df


if __name__ == "__main__":
    run()
