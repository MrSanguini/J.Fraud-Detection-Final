"""
label_builder.py, Phase C: derive per-transaction labels.

LABEL SEMANTICS (confirmed business logic)
------------------------------------------
Flag documents are APPEND-ONLY: every flag/unflag creates a NEW document, so a
transfer may have a HISTORY (flag -> unflag -> re-flag). The LATEST event by
date is the verdict. All flagged transfers are human-reviewed, so:

    latest action == "flag"    -> 1  confirmed fraud      (reviewed, upheld)
    latest action == "unflag"  -> 0  confirmed legitimate (reviewed, cleared)
    no flag document           -> 0  ASSUMED legitimate   (never caught by rules)

PRECEDENCE (highest authority first):
    1. chargeback / partner-confirmed fraud   -> 1  (EXTERNAL confirmation)
    2. flag-document verdict                  -> 1/0
    3. assumed legitimate                     -> 0

A chargeback OVERRIDES an unflag: if review cleared it but the card provider
later charged it back, that is a false negative of the review process, and the
most valuable training example available.

THE BLIND SPOT: the assumed-legit bucket contains fraud the rules never caught.
It is marked distinctly (never merged with confirmed-legit) so the model layer
can weight, exclude, or overwrite it. `coverage` in the audit quantifies this.
"""

import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from common import read_table, setup_logging, write_table

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = _P(p)
    return p if p.is_absolute() else ROOT / p


SOURCE_CHARGEBACK = "confirmed_fraud_chargeback"
SOURCE_UPHELD     = "confirmed_fraud_upheld"
SOURCE_CLEARED    = "confirmed_legit_reviewed"
SOURCE_ASSUMED    = "assumed_legit_never_flagged"


def _resolve_flag_history(flags: pd.DataFrame) -> pd.DataFrame:
    """Latest event per transfer wins. Fails loudly on shape surprises."""
    EMPTY = pd.DataFrame(columns=["_id", "label", "label_source", "verdict_date"])

    # An empty flag collection is legitimate: a period with no flags raised, or a
    # fresh deployment. Every transfer then falls through to assumed-legit.
    # Checking .empty BEFORE the column check matters -- an empty DataFrame has
    # no columns, so the schema check below would misreport it as the wrong
    # collection entirely.
    if flags is None or flags.empty:
        logger.info("no flag documents; all transfers will be assumed-legit")
        return EMPTY

    if "transfer" not in flags.columns:
        raise ValueError(
            "Flag table has no 'transfer' column. Is this the transfer-flag "
            "collection? User- and recipient-flags live in separate collections."
        )

    missing = flags["transfer"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} flag docs have a null 'transfer'. Every row in the "
            f"transfer-flag collection should reference a transfer. Investigate "
            f"before proceeding, since these would silently corrupt labels."
        )

    unexpected = set(flags["action"].dropna().unique()) - {"flag", "unflag"}
    if unexpected:
        raise ValueError(f"Unexpected action values: {unexpected}. "
                         f"Expected only 'flag' / 'unflag'.")

    f = flags.dropna(subset=["action"]).copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce", utc=True, format="mixed")
    undated = f["date"].isna().sum()
    if undated:
        raise ValueError(f"{undated} flag docs have an unparseable date. The "
                         f"append-only history cannot be ordered without them.")

    latest = (f.sort_values("date")
                .groupby("transfer", as_index=False)
                .last()[["transfer", "action", "date"]])
    latest["label"] = (latest["action"] == "flag").astype(int)
    latest["label_source"] = latest["label"].map({1: SOURCE_UPHELD, 0: SOURCE_CLEARED})
    return latest.rename(columns={"transfer": "_id", "date": "verdict_date"})


def build_labels(flags: pd.DataFrame, transfers: pd.DataFrame,
                 chargebacks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns one row per transfer: _id, label, label_source, verdict_date."""
    audit = Counter()

    latest = _resolve_flag_history(flags)
    hist = (flags.groupby("transfer").size()
            if flags is not None and not flags.empty and "transfer" in flags.columns
            else pd.Series(dtype=int))
    audit["transfers_with_flag_history"] = len(hist)
    audit["transfers_with_multiple_events"] = int((hist > 1).sum())
    audit["confirmed_fraud_upheld"] = int((latest["label"] == 1).sum())
    audit["confirmed_legit_reviewed"] = int((latest["label"] == 0).sum())

    out = transfers[["_id"]].merge(
        latest[["_id", "label", "label_source", "verdict_date"]],
        on="_id", how="left")

    orphans = set(latest["_id"]) - set(transfers["_id"])
    if orphans:
        logger.warning(f"{len(orphans)} flag docs reference transfers absent from "
                       f"the transfer table (partial export, or deleted transfers). "
                       f"Those labels are unused.")

    # ── chargebacks override everything ─────────────────────────────────────
    if chargebacks is not None and len(chargebacks):
        cb = chargebacks.dropna(subset=["_id"])[["_id", "confirmed_date"]].copy()
        cb_ids = set(cb["_id"])
        hit = out["_id"].isin(cb_ids)
        overridden = int((hit & (out["label"] == 0)).sum())
        out.loc[hit, "label"] = 1
        out.loc[hit, "label_source"] = SOURCE_CHARGEBACK
        audit["confirmed_fraud_chargeback"] = int(hit.sum())
        audit["chargebacks_overriding_a_clear"] = overridden
        unmatched = len(cb_ids - set(out["_id"]))
        if unmatched:
            logger.warning(f"{unmatched} chargebacks could not be matched to a "
                           f"transfer. Check the partner_reference join and ID formatting.")

    never = out["label"].isna()
    audit["assumed_legit_never_flagged"] = int(never.sum())
    out.loc[never, "label"] = 0
    out.loc[never, "label_source"] = SOURCE_ASSUMED
    out["label"] = out["label"].astype(int)

    # ── report ──────────────────────────────────────────────────────────────
    total = len(out)
    logger.info("[label audit]")
    for k, v in audit.items():
        logger.info(f"   {k}: {v:,}")
    logger.info(f"   total transfers: {total:,}")
    if total:
        reviewed = (out["label_source"] != SOURCE_ASSUMED).mean()
        logger.info(f"   fraud rate (as labeled): {out['label'].mean()*100:.3f}%")
        logger.info(f"   COVERAGE: {reviewed*100:.1f}% of transfers have a confirmed "
                    f"verdict; {(1-reviewed)*100:.1f}% are assumed-legit.")
        logger.info("   ^ Low coverage means the model largely learns to imitate the "
                    "existing rules. Partner/chargeback data is what fixes this.")
    return out


def run(extracted_dir=None):
    setup_logging()
    t0 = time.monotonic()
    d = _resolve(extracted_dir or config.EXTRACTED_DIR)
    try:
        flags = read_table(d / "flag")
    except FileNotFoundError:
        # extractor.py writes no flag table at all when the source has zero
        # flag documents (a fresh deployment, or an export period with none
        # raised) -- that's the realistic trigger for the crash the empty-flags
        # handling above guards against; _resolve_flag_history's `flags is None`
        # branch takes it from here.
        flags = None
        logger.info("no flag table on disk; all transfers will be assumed-legit")
    transfers = read_table(d / "transfer")
    try:
        chargebacks = read_table(d / "chargebacks")
    except FileNotFoundError:
        chargebacks = None
        logger.info("no chargeback table yet, so labels come from flag documents only.")

    labels = build_labels(flags, transfers, chargebacks)
    out = write_table(labels, d / "labels")
    logger.info(f"[write] {out}  ({len(labels):,} rows, {time.monotonic()-t0:.2f}s)")
    return labels


if __name__ == "__main__":
    run()
