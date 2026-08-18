"""
profile_schema.py — Phase A: detect schema variance BEFORE extracting.

The transfer collection is polymorphic: a field's presence can depend on the
value of a discriminator (confirmed for `paymentProvider` — 29 of 86 fields
vary across truelayer/checkout/nmi/omnex).

The extractor silently records None for absent fields, which is correct
behaviour but makes variance invisible. This script makes it visible.

RUN THIS ON EVERY FRESH EXPORT, not just once — new providers introduce new
variants over time.
"""

from collections import defaultdict
from pathlib import Path

from common import load_json, read_field
from schema_contract import CONTRACTS, DISCRIMINATORS, CANONICAL

from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent

def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = _P(p)
    return p if p.is_absolute() else ROOT / p



def profile(docs: list[dict], doc_type: str, top_n_unknown: int = 25):
    n = len(docs)
    if n == 0:
        print(f"[profile:{doc_type}] no documents")
        return {}

    presence = defaultdict(int)
    for d in docs:
        for k in d.keys():
            presence[k] += 1

    contract = CONTRACTS.get(doc_type, {})
    known = {c.split(".")[0] for c in contract}
    known |= {s for spec in CANONICAL.values() for s in spec["sources"]}

    print(f"\n{'='*70}\n[profile:{doc_type}]  {n:,} documents, "
          f"{len(presence)} distinct fields\n{'='*70}")

    # ── fields present on some docs but not others ──────────────────────────
    variable = {k: c for k, c in presence.items() if c < n}
    if variable:
        print(f"\n  VARIABLE FIELDS ({len(variable)}) — present on some docs only:")
        for k, c in sorted(variable.items(), key=lambda x: -x[1]):
            mark = "" if k in known else "   <-- NOT IN CONTRACT"
            print(f"     {k:42s} {c:>6}/{n} ({100*c/n:5.1f}%){mark}")

    # ── group by each discriminator to expose value-driven variance ─────────
    for disc in DISCRIMINATORS:
        if presence.get(disc, 0) == 0:
            continue
        groups = defaultdict(list)
        for d in docs:
            groups[str(read_field(d, disc))].append(d)
        if len(groups) < 2:
            continue
        print(f"\n  --- variance by {disc} ({len(groups)} values) ---")
        for field in sorted(variable):
            rates = {g: sum(1 for d in ds if field in d) / len(ds)
                     for g, ds in groups.items()}
            # flag fields that are ~always present in one group, ~never another
            if max(rates.values()) > 0.9 and min(rates.values()) < 0.1:
                present_in = [g for g, r in rates.items() if r > 0.9]
                print(f"     {field:38s} only when {disc} in {present_in}")

    # ── contract coverage ──────────────────────────────────────────────────
    unknown = [k for k in presence if k not in known]
    if unknown:
        print(f"\n  UNKNOWN FIELDS ({len(unknown)}) — not in contract, will be "
              f"DROPPED by allowlist:")
        for k in sorted(unknown)[:top_n_unknown]:
            print(f"     {k}")
        if len(unknown) > top_n_unknown:
            print(f"     ... and {len(unknown)-top_n_unknown} more")
        print("  Review these: anything valuable must be added to the contract "
              "explicitly.")

    # ── canonical resolution rate ──────────────────────────────────────────
    if doc_type == "transfer":
        print("\n  CANONICAL FIELD RESOLUTION:")
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
            print(f"     {canon}: {resolved}/{n} resolved ({100*resolved/n:.1f}%)")
            for src, c in hits.items():
                print(f"         via {src}: {c}")
            if unresolved:
                print(f"         !! {unresolved} UNRESOLVED — these transactions "
                      f"cannot be joined to partner chargebacks.")
    return presence


def run(data_dir="data"):
    data_dir = _resolve(data_dir)
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
        if docs:
            profile(docs, doc_type)
        else:
            print(f"[profile:{doc_type}] no files found")


if __name__ == "__main__":
    run()
