"""
extractor.py, Phase B: flatten raw JSON collections into clean flat tables.

Reads schema_contract.py and applies it mechanically. It is a DUMB FLATTENER
by design, with no feature engineering here, so it stays auditable.

Two exceptions to "dumb": it resolves CANONICAL fields (provider aliases -> one
column), and it derives sensitive values IMMEDIATELY (email -> domain) so raw
PII is never written to an intermediate file on disk.

SECURITY MODEL: exclusion, not obfuscation. Sensitive fields are DROPPED, not
hashed. Hashing is used only for non-sensitive join keys that must stay linkable.
"""

import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from common import ChunkedWriter, get_salt, hash_id, iter_documents, read_field, setup_logging
from schema_contract import CANONICAL, CONTRACTS

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


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


def extract_collection(docs: list[dict], doc_type: str) -> tuple[pd.DataFrame, Counter, Counter]:
    """Flatten one batch of documents. Pure (no I/O, no printing), so it can
    be called once per chunk without needing the whole collection in memory."""
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

    return pd.DataFrame(rows), audit, canon_sources


def run(data_dir=None, out_dir=None, limit=None, batch_size=None):
    """Streams each collection through in batches of `batch_size` documents
    (read, flatten, write, discard), so peak RAM is bounded by one batch, not
    the whole collection. Only the transfer collection is expected to reach
    millions of rows; see `common.iter_documents` for why the smaller
    auxiliary collections (users/recipients/flags) don't get the same true
    streaming under the "files" backend.

    Where documents actually come from (local export files vs. a live Mongo
    deployment) is entirely config.DATA_SOURCE's call, via
    `common.iter_documents`; this function never sees a filename or a
    collection name."""
    setup_logging()
    out_dir = _resolve(out_dir or config.EXTRACTED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or config.BATCH_SIZE

    rows_by_type = {}
    for doc_type in config.DOC_TYPES:
        t0 = time.monotonic()
        writer = ChunkedWriter(out_dir / doc_type)
        audit, canon_sources = Counter(), Counter()
        n_cols = None

        for batch in iter_documents(doc_type, data_dir=data_dir,
                                    batch_size=batch_size, limit=limit):
            df, batch_audit, batch_canon = extract_collection(batch, doc_type)
            audit.update(batch_audit)
            canon_sources.update(batch_canon)
            if n_cols is None:
                n_cols = df.shape[1]
            writer.write_batch(df)

        out = writer.close()
        if out is None:
            logger.info(f"[skip] no documents found for {doc_type}")
            continue

        elapsed = time.monotonic() - t0
        logger.info(f"[extract:{doc_type}] {audit['docs']} docs -> {n_cols} cols, "
                    f"written in batches of {batch_size} "
                    f"({audit.get('missing_tier2',0)} tier-2 field misses, expected on "
                    f"historical records)")
        if canon_sources:
            logger.info("   canonical resolution:")
            for k, v in canon_sources.most_common():
                flag = "  !! unjoinable to chargebacks" if "UNRESOLVED" in k else ""
                logger.info(f"      {k}: {v}{flag}")
        logger.info(f"[write] {out}  ({writer.rows_written} rows, {elapsed:.2f}s)")
        rows_by_type[doc_type] = writer.rows_written

    return rows_by_type


if __name__ == "__main__":
    run()
