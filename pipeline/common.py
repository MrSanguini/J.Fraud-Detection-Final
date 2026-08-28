"""
common.py: shared utilities for env loading, Mongo unwrapping, hashing, and IO.

No third-party dependencies beyond pandas. Runs entirely locally.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── logging setup ─────────────────────────────────────────────────────────────
def setup_logging(log_dir=None, level=logging.INFO) -> Path | None:
    """Configure the root logger once per process: a StreamHandler (console)
    plus a FileHandler writing to `log_dir`/pipeline_<UTC timestamp>.log; the
    per-stage row counts and timings every stage logs at INFO become a
    performance baseline you can read back after an unattended run.

    `log_dir` defaults to config.LOG_DIR.

    Idempotent: if this process already configured logging (e.g. a stage
    module run standalone after run_pipeline.py already called this), it
    returns the existing log file's path instead of adding duplicate
    handlers or starting a second file."""
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                return Path(h.baseFilename)
        return None

    log_dir = Path(log_dir or config.LOG_DIR)
    if not log_dir.is_absolute():
        log_dir = config.ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"pipeline_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info(f"logging to {log_path}")
    return log_path


# ── .env loading (minimal, no dependency) ────────────────────────────────────
def load_env(path="../.env", into_environ=True) -> dict:
    """Read KEY=value lines. Missing file raises a clear error, not silent {}."""
    p = Path(path)
    if not p.exists():
        # try project root relative to this file
        alt = Path(__file__).resolve().parent.parent / ".env"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(
                f"No .env at {p.resolve()} or {alt}. Create one containing "
                f"FRAUD_SALT=<long random string>. It must be gitignored."
            )
    values = {}
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f".env line {lineno} is not KEY=value: {raw!r}")
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        values[key] = val
        if into_environ:
            os.environ.setdefault(key, val)
    return values


def get_salt() -> bytes:
    salt = os.environ.get("FRAUD_SALT")
    if not salt:
        salt = load_env().get("FRAUD_SALT")
    if not salt:
        raise RuntimeError("FRAUD_SALT missing. Set it in .env; never hardcode it.")
    return salt.encode("utf-8")


# ── Mongo export handling ────────────────────────────────────────────────────
def unwrap(v):
    """Mongo wraps values: {'$oid':..}, {'$date':..}, {'$numberLong':..}.
    Return the plain scalar underneath. Handles nested $date/$numberLong."""
    if isinstance(v, dict):
        if "$oid" in v:
            return v["$oid"]
        if "$date" in v:
            d = v["$date"]
            return unwrap(d) if isinstance(d, dict) else d
        if "$numberLong" in v:
            return int(v["$numberLong"])
        if "$numberInt" in v:
            return int(v["$numberInt"])
        if "$numberDouble" in v:
            return float(v["$numberDouble"])
    return v


def get_nested(doc: dict, dotted: str):
    """Fetch by dotted path ('ipDetails.location.city'), tolerating gaps."""
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def read_field(doc: dict, field: str):
    """Read a possibly-nested field and unwrap Mongo types. '' -> None."""
    raw = get_nested(doc, field) if "." in field else doc.get(field)
    val = unwrap(raw)
    if isinstance(val, str) and val.strip() == "":
        return None          # empty strings (e.g. account_id) are missing values
    return val


# ── pseudonymisation ─────────────────────────────────────────────────────────
def hash_id(value, salt: bytes) -> str | None:
    """Deterministic salted SHA-256 pseudonym.

    Same input + same salt -> same token, so joins survive. Without the salt it
    is not reversible. NEVER used as a model feature: join plumbing only.
    Values are normalised (str, stripped, lowercased) so that the same identity
    written slightly differently still matches."""
    value = unwrap(value)
    if value is None:
        return None
    norm = str(value).strip().lower()
    if not norm:
        return None
    return hashlib.sha256(salt + norm.encode("utf-8")).hexdigest()[:24]


# ── JSON loading ─────────────────────────────────────────────────────────────
def load_json(path, limit: int | None = None) -> list[dict]:
    """Load a JSON array or NDJSON of documents.

    Tolerates whole-line // and /* */ comments (mock exports sometimes have
    them). Deliberately does NOT strip mid-line '//' because that appears
    inside legitimate string values such as URLs.

    `limit`, when given, caps how many documents are returned from THIS file
    (the JSON-file equivalent of a pymongo cursor's .limit(N)), for cheap
    smoke runs against a small slice of a source instead of the whole export."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"(?m)^\s*//.*$", "", no_block)
    try:
        if cleaned.lstrip().startswith("["):
            docs = json.loads(cleaned)
        else:
            docs = [json.loads(l) for l in cleaned.splitlines() if l.strip()]
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse {p} as JSON at line {e.lineno}, col {e.colno}: "
            f"{e.msg}. Expect a JSON array or newline-delimited JSON."
        ) from e
    return docs[:limit] if limit is not None else docs


def _sniff_is_array(p: Path) -> bool:
    """Peek past leading blank/comment lines to see if the file is a JSON
    array (needs a full parse) or NDJSON (streamable)."""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            return s.startswith("[")
    return False


def iter_json_batches(path, batch_size: int = 10000, limit: int | None = None):
    """Yield lists of up to `batch_size` documents from `path`.

    NDJSON files are streamed line-by-line: at most one batch is ever held
    in memory, which is what keeps peak RAM flat as a source collection
    grows into the millions. JSON-array files (the smaller auxiliary
    collections, users/recipients/flags, which don't scale anywhere near
    as fast as transfers) can't be parsed incrementally without a streaming
    JSON library, so they're loaded once via `load_json` and re-sliced into
    batches; that's a fine trade since it's the transfer collection this
    guards against, not those.

    `limit` caps the TOTAL documents yielded from this file, mirroring a
    pymongo cursor's .limit(N)."""
    p = Path(path)
    emitted = 0

    def _cap(batch: list[dict]) -> list[dict]:
        nonlocal emitted
        if limit is not None:
            batch = batch[:max(0, limit - emitted)]
        emitted += len(batch)
        return batch

    if _sniff_is_array(p):
        docs = load_json(p)
        for i in range(0, len(docs), batch_size):
            batch = _cap(docs[i:i + batch_size])
            if not batch:
                return
            yield batch
        return

    batch: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = re.sub(r"/\*.*?\*/", "", raw)
            line = re.sub(r"^\s*//.*$", "", line).strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse a line of {p} as JSON: {e.msg}. Expect "
                    f"newline-delimited JSON, one document per line."
                ) from e
            if len(batch) >= batch_size:
                capped = _cap(batch)
                batch = []
                if not capped:
                    return
                yield capped
                if limit is not None and emitted >= limit:
                    return
    if batch:
        capped = _cap(batch)
        if capped:
            yield capped


def _iter_mongo_batches(doc_type: str, batch_size: int, limit: int | None):
    """Yield batches of documents from a live Mongo collection via a
    server-side cursor, so peak RAM stays bounded exactly like
    `iter_json_batches` (DATA_SOURCE == "atlas").

    Extended-JSON wrapping ({'$oid': ...}, {'$date': ...}) doesn't apply
    here: pymongo already deserialises BSON into native types (ObjectId,
    datetime), and `unwrap()` passes anything that isn't a wrapped dict
    through unchanged, so downstream code (read_field, hash_id, pandas
    date parsing) handles native pymongo documents with no extra work."""
    import pymongo  # optional dependency: `pip install pymongo`

    collection_name = config.MONGO_COLLECTIONS.get(doc_type)
    if not collection_name:
        raise ValueError(f"No Mongo collection configured for doc_type {doc_type!r} "
                         f"in config.MONGO_COLLECTIONS.")
    if not config.MONGO_URI:
        raise RuntimeError("config.MONGO_URI is not set. Export it as an environment "
                           "variable before running with DATA_SOURCE=atlas.")

    client = pymongo.MongoClient(config.MONGO_URI)
    try:
        cursor = client[config.MONGO_DB][collection_name].find(batch_size=batch_size)
        if limit is not None:
            cursor = cursor.limit(limit)
        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        client.close()


def iter_documents(doc_type: str, data_dir=None, batch_size: int | None = None,
                   limit: int | None = None):
    """Yield batches of raw documents for `doc_type`, from whichever backend
    config.DATA_SOURCE selects. This is the ONLY place stage modules should
    read source data from: extractor.py and profile_schema.py never touch a
    filename or a Mongo collection name directly, so pointing the pipeline
    at a different source is a config.py change, not a code change."""
    batch_size = batch_size or config.BATCH_SIZE

    if config.DATA_SOURCE == "files":
        data_dir = Path(data_dir or config.DATA_DIR)
        if not data_dir.is_absolute():
            data_dir = config.ROOT / data_dir
        for fn in config.LOCAL_COLLECTIONS.get(doc_type, []):
            p = data_dir / fn
            if p.exists():
                yield from iter_json_batches(p, batch_size=batch_size, limit=limit)
    elif config.DATA_SOURCE == "atlas":
        yield from _iter_mongo_batches(doc_type, batch_size=batch_size, limit=limit)
    else:
        raise ValueError(f"Unknown config.DATA_SOURCE {config.DATA_SOURCE!r}; "
                         f"expected 'files' or 'atlas'.")


# ── tabular IO with graceful parquet/CSV fallback ────────────────────────────
def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, path_no_ext: Path) -> Path:
    """Write parquet when available (preserves dtypes), else CSV."""
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if _has_parquet():
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


class ChunkedWriter:
    """Streams DataFrame batches to a single output file, holding at most one
    batch in memory: the write-side counterpart to `iter_json_batches`.

    Parquet: appends via pyarrow.parquet.ParquetWriter. Each batch is cast to
    the schema established by the first (safe for things like an all-null
    column inferring a different dtype in a later batch); a genuine type
    conflict raises rather than silently corrupting the file. CSV: appends
    rows, writing the header once. Call close() when done."""

    def __init__(self, path_no_ext: Path):
        self.path_no_ext = Path(path_no_ext)
        self.path_no_ext.parent.mkdir(parents=True, exist_ok=True)
        self._use_parquet = _has_parquet()
        self._writer = None      # pyarrow ParquetWriter, opened on first batch
        self._started = False
        self._path = None
        self._rows = 0

    def write_batch(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if self._use_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._writer is None:
                self._path = self.path_no_ext.with_suffix(".parquet")
                self._writer = pq.ParquetWriter(self._path, table.schema)
            else:
                table = table.cast(self._writer.schema)
            self._writer.write_table(table)
        else:
            self._path = self.path_no_ext.with_suffix(".csv")
            df.to_csv(self._path, mode="a" if self._started else "w",
                     header=not self._started, index=False)
        self._started = True
        self._rows += len(df)

    def close(self) -> Path | None:
        """Returns the output path, or None if no batch was ever written."""
        if self._writer is not None:
            self._writer.close()
        return self._path if self._started else None

    @property
    def rows_written(self) -> int:
        return self._rows


def read_table(path_no_ext: Path) -> pd.DataFrame:
    """Read whichever of .parquet/.csv exists."""
    path_no_ext = Path(path_no_ext)
    pq, csv = path_no_ext.with_suffix(".parquet"), path_no_ext.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pq} nor {csv} exists. Run the earlier stage first.")


def table_exists(path_no_ext: Path) -> bool:
    """True if a table (parquet or CSV form) already exists at this path,
    which run_pipeline.py uses to decide whether a stage's output is already
    checkpointed and the stage can be skipped."""
    path_no_ext = Path(path_no_ext)
    return (path_no_ext.with_suffix(".parquet").exists()
            or path_no_ext.with_suffix(".csv").exists())


# ── git checkpoint ───────────────────────────────────────────────────────────
def checkpoint(message: str, repo_dir=".", push: bool = True) -> None:
    """Stage all, commit, push. Never raises on an empty commit; a failed push
    warns rather than losing the local commit."""
    repo = Path(repo_dir).resolve()

    def run(args, check=True):
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
        return r

    run(["add", "-A"])
    c = run(["commit", "-m", message], check=False)
    if c.returncode != 0:
        if "nothing to commit" in (c.stdout + c.stderr).lower():
            logger.info("checkpoint: nothing to commit, skipping.")
            return
        raise RuntimeError(f"git commit failed:\n{c.stderr.strip() or c.stdout.strip()}")
    logger.info(f"checkpoint: committed, {message!r}")
    if push:
        p = run(["push"], check=False)
        if p.returncode == 0:
            logger.info("checkpoint: pushed.")
        else:
            logger.warning(f"checkpoint: commit saved locally but push failed:\n{p.stderr.strip()}")
