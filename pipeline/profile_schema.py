"""
profile_schema.py, Phase A: detect schema variance BEFORE extracting.

The transfer collection is polymorphic: a field's presence can depend on the
value of a discriminator (confirmed for `paymentProvider`: 29 of 86 fields
vary across truelayer/checkout/nmi/omnex).

The extractor silently records None for absent fields, which is correct
behaviour but makes variance invisible. This script makes it visible.

RUN THIS ON EVERY FRESH EXPORT, not just once, since new providers introduce
new variants over time.
"""

import logging
import time
from collections import defaultdict

import config
from common import iter_documents, read_field, setup_logging
from schema_contract import CONTRACTS, DISCRIMINATORS, CANONICAL

logger = logging.getLogger(__name__)


def profile(docs: list[dict], doc_type: str, top_n_unknown: int = 25):
    n = len(docs)
    if n == 0:
        logger.info(f"[profile:{doc_type}] no documents")
        return {}

    presence = defaultdict(int)
    for d in docs:
        for k in d.keys():
            presence[k] += 1

    contract = CONTRACTS.get(doc_type, {})
    known = {c.split(".")[0] for c in contract}
    known |= {s for spec in CANONICAL.values() for s in spec["sources"]}

    logger.info(f"[profile:{doc_type}] {n:,} documents, {len(presence)} distinct fields")

    # ── fields present on some docs but not others ──────────────────────────
    variable = {k: c for k, c in presence.items() if c < n}
    if variable:
        logger.info(f"  VARIABLE FIELDS ({len(variable)}), present on some docs only:")
        for k, c in sorted(variable.items(), key=lambda x: -x[1]):
            mark = "" if k in known else "   <-- NOT IN CONTRACT"
            logger.info(f"     {k:42s} {c:>6}/{n} ({100*c/n:5.1f}%){mark}")

    # ── group by each discriminator to expose value-driven variance ─────────
    for disc in DISCRIMINATORS:
        if presence.get(disc, 0) == 0:
            continue
        groups = defaultdict(list)
        for d in docs:
            groups[str(read_field(d, disc))].append(d)
        if len(groups) < 2:
            continue
        logger.info(f"  --- variance by {disc} ({len(groups)} values) ---")
        for field in sorted(variable):
            rates = {g: sum(1 for d in ds if field in d) / len(ds)
                     for g, ds in groups.items()}
            # flag fields that are ~always present in one group, ~never another
            if max(rates.values()) > 0.9 and min(rates.values()) < 0.1:
                present_in = [g for g, r in rates.items() if r > 0.9]
                logger.info(f"     {field:38s} only when {disc} in {present_in}")

    # ── contract coverage ──────────────────────────────────────────────────
    unknown = [k for k in presence if k not in known]
    if unknown:
        logger.info(f"  UNKNOWN FIELDS ({len(unknown)}), not in contract, will be "
                    f"DROPPED by allowlist:")
        for k in sorted(unknown)[:top_n_unknown]:
            logger.info(f"     {k}")
        if len(unknown) > top_n_unknown:
            logger.info(f"     ... and {len(unknown)-top_n_unknown} more")
        logger.info("  Review these: anything valuable must be added to the contract "
                    "explicitly.")

    # ── canonical resolution rate ──────────────────────────────────────────
    if doc_type == "transfer":
        logger.info("  CANONICAL FIELD RESOLUTION:")
        for canon, spec in CANONICAL.items():
            hits = defaultdict(int)
            unresolved = 0
            for d in docs:
                for src in spec["sources"]:
                    if read_field(d, src) is not None:
                        hits[src] += 1
                        break
                else:
                    unresolved += 1
            resolved = n - unresolved
            logger.info(f"     {canon}: {resolved}/{n} resolved ({100*resolved/n:.1f}%)")
            for src, c in hits.items():
                logger.info(f"         via {src}: {c}")
            if unresolved:
                logger.warning(f"     {canon}: {unresolved} UNRESOLVED: these "
                               f"transactions cannot be joined to partner chargebacks.")
    return presence


def run(data_dir=None, limit=None):
    """Profiling needs every document in memory at once to compute
    field-presence rates, so (unlike extractor.run) this doesn't stream in
    fixed batches; it still reads via `common.iter_documents` in whatever
    chunk size config.BATCH_SIZE gives it, so the source (files vs. a live
    Mongo deployment) is still entirely config.DATA_SOURCE's call."""
    setup_logging()
    counts = {}
    for doc_type in config.DOC_TYPES:
        t0 = time.monotonic()
        docs = []
        for batch in iter_documents(doc_type, data_dir=data_dir, limit=limit):
            docs.extend(batch)
        if docs:
            profile(docs, doc_type)
        else:
            logger.info(f"[profile:{doc_type}] no documents found")
        counts[doc_type] = len(docs)
        logger.info(f"[profile:{doc_type}] {len(docs):,} docs profiled in "
                    f"{time.monotonic()-t0:.2f}s")
    return counts


if __name__ == "__main__":
    run()
