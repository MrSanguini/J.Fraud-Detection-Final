"""
common.py — shared utilities: env loading, Mongo unwrapping, hashing, IO.

No third-party dependencies beyond pandas. Runs entirely locally.
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


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
        raise RuntimeError("FRAUD_SALT missing. Set it in .env — never hardcode it.")
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
    is not reversible. NEVER used as a model feature — join plumbing only.
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
def load_json(path) -> list[dict]:
    """Load a JSON array or NDJSON of documents.

    Tolerates whole-line // and /* */ comments (mock exports sometimes have
    them). Deliberately does NOT strip mid-line '//' because that appears
    inside legitimate string values such as URLs."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"(?m)^\s*//.*$", "", no_block)
    try:
        if cleaned.lstrip().startswith("["):
            return json.loads(cleaned)
        return [json.loads(l) for l in cleaned.splitlines() if l.strip()]
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse {p} as JSON at line {e.lineno}, col {e.colno}: "
            f"{e.msg}. Expect a JSON array or newline-delimited JSON."
        ) from e


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


def read_table(path_no_ext: Path) -> pd.DataFrame:
    """Read whichever of .parquet/.csv exists."""
    path_no_ext = Path(path_no_ext)
    pq, csv = path_no_ext.with_suffix(".parquet"), path_no_ext.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pq} nor {csv} exists. Run the earlier stage first.")


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
            print("checkpoint: nothing to commit — skipping.")
            return
        raise RuntimeError(f"git commit failed:\n{c.stderr.strip() or c.stdout.strip()}")
    print(f"checkpoint: committed — {message!r}")
    if push:
        p = run(["push"], check=False)
        print("checkpoint: pushed." if p.returncode == 0
              else f"checkpoint: commit saved locally but push failed:\n{p.stderr.strip()}")
