"""
extractor.py — Phase B: flatten raw JSON collections into clean flat tables.

Reads schema_contract.py and applies it mechanically. It is a DUMB FLATTENER by
design — no feature engineering here, so it stays auditable.

Two exceptions to "dumb": it resolves CANONICAL fields (provider aliases -> one
column), and it derives sensitive values IMMEDIATELY (email -> domain) so raw
PII is never written to an intermediate file on disk.

SECURITY MODEL: exclusion, not obfuscation. Sensitive fields are DROPPED, not
hashed. Hashing is used only for non-sensitive join keys that must stay linkable.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from common import get_salt, hash_id, load_json, read_field, write_table
from schema_contract import CANONICAL, CONTRACTS

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = _P(p)
    return p if p.is_absolute() else ROOT / p


SALT = get_salt()


# ── sensitive DERIVE fields: transformed at read time, raw never persisted ───
def email_domain(v):
    """Domain only. The local part (the person) is discarded immediately."""
    if not isinstance(v, str) or "@" not in v:
        return None
    return v.rsplit("@", 1)[1].strip().lower()


SENSITIVE_DERIVE = {
    # field -> (output column, transform). None output = drop entirely.
    "sender_email":    ("sender_email_domain", email_domain),
    "recipient_email": ("recipient_email_domain", email_domain),
    "ipDetails.ip":    (None, None),   # geo already resolved into country/city
    "billing_address": ("_raw_billing_address", lambda v: v),  # tier-2, consumed by assembler
}


def resolve_canonical(doc: dict, spec: dict):
    """First non-null among aliased source fields. Returns (value, which_source)."""
    for src in spec["sources"]:
        val = read_field(doc, src)
        if val is not None:
            return val, src
    return None, None


def extract_collection(docs: list[dict], doc_type: str) -> pd.DataFrame:
    contract = CONTRACTS[doc_type]
    rows, audit = [], Counter()
    canon_sources = Counter()

    for doc in docs:
        audit["docs"] += 1
        row = {}

        for field, (action, _sens, _mut, tier, _reason) in contract.items():
            if action == "DROP":
                continue
            raw = read_field(doc, field)
            col = field.replace(".", "__")

            if raw is None:
                audit[f"missing_tier{tier}"] += 1

            if action == "HASH":
                row[col] = hash_id(raw, SALT)
            elif action == "DERIVE":
                if field in SENSITIVE_DERIVE:
                    out_col, fn = SENSITIVE_DERIVE[field]
                    if out_col:
                        row[out_col] = fn(raw) if raw is not None else None
                    # out_col None -> written nowhere; raw value discarded
                else:
                    row[f"_raw_{col}"] = raw
            else:                                    # KEEP, LABEL, META
                row[col] = raw

        # canonical (provider-variant) fields
        if doc_type == "transfer":
            for canon, spec in CANONICAL.items():
                val, src = resolve_canonical(doc, spec)
                row[canon] = hash_id(val, SALT) if spec["action"] == "HASH" else val
                if src:
                    canon_sources[f"{canon}<-{src}"] += 1
                else:
                    canon_sources[f"{canon}<-UNRESOLVED"] += 1

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[extract:{doc_type}] {audit['docs']} docs -> {df.shape[1]} cols "
          f"({audit.get('missing_tier2',0)} tier-2 field misses, expected on "
          f"historical records)")
    if canon_sources:
        print("   canonical resolution:")
        for k, v in canon_sources.most_common():
            flag = "  !! unjoinable to chargebacks" if "UNRESOLVED" in k else ""
            print(f"      {k}: {v}{flag}")
    return df


def run(data_dir="data", out_dir="data/extracted"):
    data_dir, out_dir = _resolve(data_dir), _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multiple files per collection are concatenated — this is how provider
    # variants (which may be exported separately) get unified.
    collections = {
        "transfer":  ["transfers.json", "transfer.json", "checkout_transfers.json",
                      "nmi_transfers.json", "omnex_transfers.json"],
        "user":      ["users.json", "user.json"],
        "recipient": ["recipients.json"],
        "flag":      ["transfer_flags.json", "flags.json"],
    }

    for doc_type, filenames in collections.items():
        docs = []
        for fn in filenames:
            p = data_dir / fn
            if p.exists():
                docs.extend(load_json(p))
        if not docs:
            print(f"[skip] no files for {doc_type}")
            continue
        df = extract_collection(docs, doc_type)
        out = write_table(df, out_dir / doc_type)
        print(f"[write] {out}  ({len(df)} rows)")


if __name__ == "__main__":
    run()
