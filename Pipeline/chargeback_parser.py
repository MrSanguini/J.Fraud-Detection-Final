"""
chargeback_parser.py — Phase C0: parse partner chargeback sheets into labels.

Partner-confirmed fraud is the ONLY source of "fraud the rules never caught".
Without it the model learns to imitate the existing rules engine; with it, the
model can improve on it.

INPUT: a CSV/XLSX export from the partner (NOT screenshots — ask partners for
the underlying file; transcribing digits from images silently corrupts labels,
and a label error teaches the model that a legitimate transaction was fraud).

Observed sheet columns:
    AGENT_CODE, AGENT_DBA, DEPOSIT_DATE, AMOUNT, DEPOSIT_NOTE,
    UNIQUE TRNX ID @<partner>, FOLIO, SENDER_NAME/MIDDLENAME/LASTNAME,
    SENDING_PRINCIPAL, SENDING_DATE, RB_DATE, Observations // Patterns,
    Watch List / WL

USE AS LABELS, NOT FEATURES. Almost every column is post-hoc — it exists only
BECAUSE a chargeback happened, weeks after the transaction. Feeding any of it to
the model is textbook target leakage. We keep only:
    FOLIO / TRNX ID  -> join keys (hashed with the SAME salt as the transfers)
    RB_DATE          -> confirmation date (metadata for point-in-time logic)
    AMOUNT           -> reconciliation only
    Observations     -> coarse fraud_type, for per-typology REPORTING only
"""

import re
from pathlib import Path

import pandas as pd

from common import get_salt, hash_id, write_table

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = _P(p)
    return p if p.is_absolute() else ROOT / p


SALT = get_salt()

# tolerate header drift across partners/exports
COLUMN_ALIASES = {
    "folio":        ["folio", "folio (trnx id at jupay)", "trnx id at jupay",
                     "common id", "partner_reference"],
    "partner_txn":  ["unique trnx id", "trnx id", "partner txn id", "unique trnx id @mtn"],
    "amount":       ["amount", "chargeback amount"],
    "principal":    ["sending_principal", "sending principal"],
    "sending_date": ["sending_date", "sending date"],
    "rb_date":      ["rb_date", "rb date", "chargeback date"],
    "observations": ["observations // patterns", "observations", "patterns", "notes"],
    "reason_code":  ["reason_code", "reason code", "error code"],
    "watchlist":    ["watch list / wl", "watchlist", "wl"],
}

# coarse typology from free-text observations — for REPORTING, never a feature
TYPOLOGY_PATTERNS = [
    ("stolen_card",       r"stolen card|lost card|card.*stolen"),
    ("no_authorization",  r"no cardholder auth|unauthori[sz]ed"),
    ("friendly_fraud",    r"friendly fraud|first.?party|not recogni[sz]ed"),
    ("account_takeover",  r"account takeover|ato\b|compromised account"),
    ("synthetic_identity",r"no id in system|new customer.*no id|synthetic"),
    ("scam_app",          r"scam|romance|impersonat|social engineer"),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _find_column(df: pd.DataFrame, key: str) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        for norm, orig in norm_map.items():
            if alias in norm:
                return orig
    return None


def classify_typology(text) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.lower()
    for label, pattern in TYPOLOGY_PATTERNS:
        if re.search(pattern, t):
            return label
    return "other"


def parse_chargebacks(df: pd.DataFrame) -> pd.DataFrame:
    """Map a partner sheet to a label table. Hashes join keys with the pipeline
    salt so they match the extractor's hashed partner_reference."""
    folio_col = _find_column(df, "folio")
    txn_col = _find_column(df, "partner_txn")
    if folio_col is None and txn_col is None:
        raise ValueError(
            f"No join column found. Need a FOLIO or partner-transaction-ID "
            f"column. Saw: {list(df.columns)}"
        )

    out = pd.DataFrame(index=df.index)
    # hash BOTH keys — extractor emits partner_reference and partner_txn_id
    out["partner_reference"] = (df[folio_col].apply(lambda v: hash_id(v, SALT))
                                if folio_col else None)
    out["partner_txn_id"] = (df[txn_col].apply(lambda v: hash_id(v, SALT))
                             if txn_col else None)

    rb = _find_column(df, "rb_date")
    out["confirmed_date"] = (pd.to_datetime(df[rb], errors="coerce", utc=True)
                             if rb else pd.NaT)

    amt = _find_column(df, "amount")
    # chargebacks are debits, often negative — compare magnitudes
    out["chargeback_amount"] = (pd.to_numeric(df[amt], errors="coerce").abs()
                                if amt else pd.NA)

    obs = _find_column(df, "observations")
    code = _find_column(df, "reason_code")
    if code:
        out["fraud_type"] = df[code].astype("string")     # structured beats prose
    elif obs:
        out["fraud_type"] = df[obs].apply(classify_typology)
        print("[info] no structured reason code — typology inferred from free "
              "text. Ask partners for the raw reason code; it is far more reliable.")
    else:
        out["fraud_type"] = None

    unresolved = out["partner_reference"].isna() & out["partner_txn_id"].isna()
    if unresolved.any():
        print(f"[WARN] {unresolved.sum()} chargeback rows have no usable join key.")

    print(f"[chargebacks] parsed {len(out)} rows")
    if out["fraud_type"].notna().any():
        print("   typology mix:")
        for k, v in out["fraud_type"].value_counts().items():
            print(f"      {k}: {v}")
    return out


def match_to_transfers(cb: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Resolve chargebacks to transfer _ids via either hashed key."""
    matched = pd.Series(pd.NA, index=cb.index, dtype="object")

    for key in ["partner_reference", "partner_txn_id"]:
        if key not in transfers.columns or key not in cb.columns:
            continue
        lookup = (transfers.dropna(subset=[key])
                           .drop_duplicates(subset=[key])
                           .set_index(key)["_id"])
        fill = cb[key].map(lookup)
        matched = matched.fillna(fill)

    out = cb.copy()
    out["_id"] = matched
    rate = out["_id"].notna().mean() * 100 if len(out) else 0
    print(f"[chargebacks] matched {out['_id'].notna().sum()}/{len(out)} "
          f"({rate:.1f}%) to transfers")
    if rate < 90:
        print("   [WARN] low match rate. Check: (a) ID formatting/prefixes differ "
              "between partner sheet and our records, (b) some providers write no "
              "partner reference at all — those chargebacks are unmatchable and "
              "create label-coverage bias.")
    return out


def run(sheet_path, extracted_dir="data/extracted"):
    p = _resolve(sheet_path)
    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    cb = parse_chargebacks(df)
    from common import read_table
    transfers = read_table(_resolve(extracted_dir) / "transfer")
    cb = match_to_transfers(cb, transfers)
    out = write_table(cb, _resolve(extracted_dir) / "chargebacks")
    print(f"[write] {out}")
    return cb


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python chargeback_parser.py <partner_sheet.csv|xlsx>")
    else:
        run(sys.argv[1])
