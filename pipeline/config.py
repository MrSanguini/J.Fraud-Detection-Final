"""
config.py: single source of truth for pipeline-wide settings: data paths,
collection names, batch sizing, feature windows, and cost constants.

Every stage module reads from here instead of hardcoding these values, so
pointing the pipeline at a different data source, tuning batch size, or
adjusting a business constant is a config change, not a code change.

DECLARATIVE ONLY, like schema_contract.py: nothing here reads or transforms
pipeline data. The one exception is MONGO_URI/MONGO_DB, read from the
process environment (never hardcoded here, since a connection string can
carry credentials).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where pipeline OUTPUT (extracted tables, training_set, artifacts, logs) is
# written. Defaults to the project root, same as always. Override via
# PIPELINE_OUTPUT_ROOT to redirect ALL of it elsewhere in one move -- e.g.
# the test suite points this at testing/test_outputs/ so running pytest
# never touches the real data/, artifacts/, or logs/ directories.
#
# Does NOT affect where SOURCE data is read from (DATA_DIR below, or
# run_pipeline.py's --data-dir): a test suite already controls that
# separately by pointing --data-dir at its own synthetic export.
OUTPUT_ROOT = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", ROOT))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = ROOT / OUTPUT_ROOT


# ── data source ──────────────────────────────────────────────────────────
# "files": read local JSON/NDJSON exports from DATA_DIR (the only backend
#          exercised against real data today).
# "atlas": read live from a MongoDB Atlas (or any Mongo) deployment via
#          MONGO_URI/MONGO_DB, using MONGO_COLLECTIONS for collection names.
#          Requires `pip install pymongo` (not a hard dependency of this
#          project, since most environments only ever use "files").
#
# Stage modules never open a file or a Mongo connection themselves; they
# call common.iter_documents(doc_type, ...), which reads this flag. That
# indirection is what makes switching backends a one-line change here
# instead of edits to extractor.py / profile_schema.py.
DATA_SOURCE = os.environ.get("PIPELINE_DATA_SOURCE", "files")

# Mongo connection settings. Set MONGO_URI as a real environment variable,
# e.g. `export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net"`;
# unlike FRAUD_SALT this is NOT auto-loaded from .env, because config.py is
# imported before other modules load .env and reading it here would be
# order-dependent. Irrelevant when DATA_SOURCE == "files".
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB", "production")

# canonical collection identity used throughout the pipeline
DOC_TYPES = ("transfer", "user", "recipient", "flag")

# doc_type -> Mongo collection/view name (DATA_SOURCE == "atlas")
MONGO_COLLECTIONS = {
    "transfer":  "transfers",
    "user":      "users",
    "recipient": "recipients",
    "flag":      "transfer_flags",
}

# doc_type -> candidate local export filenames, tried in order and
# concatenated (DATA_SOURCE == "files"). Multiple filenames are how
# provider variants exported as separate files get unified into one
# collection.
LOCAL_COLLECTIONS = {
    "transfer":  ["transfers.json", "transfer.json", "checkout_transfers.json",
                  "nmi_transfers.json", "omnex_transfers.json"],
    "user":      ["users.json", "user.json"],
    "recipient": ["recipients.json"],
    "flag":      ["transfer_flags.json", "flags.json"],
}


# ── paths ────────────────────────────────────────────────────────────────
# DATA_DIR is the SOURCE default (relative to ROOT, not OUTPUT_ROOT) --
# run_pipeline.py's --data-dir overrides it per-invocation, same as always.
DATA_DIR = "data"

# Everything below is OUTPUT, anchored at OUTPUT_ROOT and therefore already
# absolute. Every module resolves paths via `p if p.is_absolute() else
# ROOT / p` (or pathlib's own "absolute operand wins" behaviour for the two
# `ROOT / config.X` spots in run_pipeline.py), so these pass through
# unchanged wherever OUTPUT_ROOT != ROOT -- no other file needs to change
# for the redirect to take effect.
EXTRACTED_DIR = OUTPUT_ROOT / "data" / "extracted"
TRAINING_SET_PATH = OUTPUT_ROOT / "data" / "training_set"
LOG_DIR = OUTPUT_ROOT / "logs"
ARTIFACTS_DIR = OUTPUT_ROOT / "artifacts"    # model_prep.py: fitted ModelPrep, saved via joblib


# ── processing ──────────────────────────────────────────────────────────
BATCH_SIZE = 10000          # documents per chunk in the extractor (RAM/speed tradeoff)
SEED = 42                   # train.py: every random_state/seed argument, for reproducibility


# ── behavioural feature windows ─────────────────────────────────────────
# Rolling velocity windows, in hours, used by add_velocity_features() in
# assembler.py: transaction count/sum over the trailing N hours per user.
VELOCITY_WINDOWS_HOURS = (1, 24, 168)


# ── cost model ────────────────────────────────────────────────────────────
# Consumed by metrics.py's cost() and train.py's threshold selection.
# All three are PLACEHOLDERS, not confirmed with finance/ops -- metrics.py's
# own docstring says fp_unit_cost is "NOT measurable from the data" by
# design, so these are a starting point for that conversation, not a
# figure to trust as-is.
FP_UNIT_COST = 25.00        # $ cost of one wrongly-declined legitimate customer
FN_LOSS_FRACTION = 1.0      # fraction of a missed fraud's amount the business eats
FN_FIXED_COST = 0.0         # $ per-fraud overhead independent of amount (chargeback fees, etc.)
