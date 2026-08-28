"""
model_prep.py: the bridge between the pipeline's training_set and the model.

The pipeline produces a clean, pseudonymised table. It does NOT produce a
model-ready feature matrix. Three things stand between them:

  1. TEMPORAL SPLIT      -- train on the past, test on the future. Never shuffle.
  2. FREQUENCY ENCODING  -- high-cardinality categoricals (bank names, cities,
                            ASNs) are useless raw; rarity itself is the signal.
  3. CATEGORICAL DTYPES  -- so XGBoost's enable_categorical=True works, with the
                            levels pinned from TRAIN so unseen values are handled.

LEAKAGE DISCIPLINE
------------------
Everything here is FIT ON TRAIN ONLY and merely applied to val/test. The
metadata columns are excluded from features by construction:

  label         the target
  label_source  PERFECT LEAKAGE -- maps 1:1 to the label. A model given this
                would score ~100% and be worthless. Excluded, never optional.
  date          drives the split; `hour`/`weekday` are the usable derivatives
  amount_paid   the realised-loss figure for the cost model. Not knowable at
                scoring time, so it is a metric input, never a feature.

CURRENT GAP: assembler.py drops amount_paid before writing training_set (and
test_pipeline.py's LEAKAGE_COLUMNS actively fails a run if it survives), so
AMOUNT_COL is never actually present today -- `prepare()` below always falls
back to send_amount (the attempted amount, not the realised loss). That's a
graceful degradation, not a crash, but it means the cost-model amounts this
returns are NOT what this module's own docstring describes. Whether
amount_paid is safe to carry as metadata-only (excluded from features, same
as label_source) or is genuine leakage under this business's scoring-time
semantics is a call for whoever owns that assessment, not something to
silently flip here.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from common import read_table, setup_logging

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _resolve(p):
    """Resolve a path against the project root unless already absolute."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _is_bool_like(series: pd.Series) -> bool:
    """True for real bool dtype, and also for an object-dtype column whose
    only non-null values are Python bools -- which is what a JSON-sourced
    boolean column with some missing rows becomes (pandas can't hold NaN in
    a numpy bool array, so it falls back to object dtype). Both need
    numeric passthrough, not categorical encoding: XGBoost's categorical
    index rejects raw True/False values ("Category index must contain only
    values of the same type, either string or integer"). Deliberately
    checks isinstance(..., bool) rather than `.isin([True, False])`, since
    the latter also matches a genuine 0/1 integer column in Python."""
    if series.dtype == bool:
        return True
    non_null = series.dropna()
    return len(non_null) > 0 and non_null.map(lambda v: isinstance(v, (bool, np.bool_))).all()


TARGET = "label"
TIME_COL = "date"
AMOUNT_COL = "amount_paid"          # realised loss, for the cost model only

# Columns that exist in the training set but must NEVER enter the feature matrix.
METADATA_COLS = [TARGET, "label_source", TIME_COL, AMOUNT_COL]


# ── temporal split ───────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame, test_frac=0.20, val_frac=0.15,
                   time_col=TIME_COL):
    """Split by TIME, not at random.

    train | val | test, ordered chronologically. The val slice is the tail of
    the training period and is used for early stopping, model selection, and
    threshold choice -- so the test set stays untouched until a single final read.

    A random split would scatter future transactions into training and inflate
    every metric. There is no random_state here because there is no randomness.
    """
    d = df.copy()
    t = pd.to_datetime(d[time_col], errors="coerce", utc=True, format="mixed")
    bad = t.isna().sum()
    if bad:
        raise ValueError(f"{bad} rows have an unparseable {time_col}; cannot order them.")
    d = d.assign(_ts=t).sort_values("_ts").reset_index(drop=True)

    n = len(d)
    n_test = int(n * test_frac)
    n_val = int((n - n_test) * val_frac)
    n_train = n - n_test - n_val

    train = d.iloc[:n_train].drop(columns="_ts")
    val = d.iloc[n_train:n_train + n_val].drop(columns="_ts")
    test = d.iloc[n_train + n_val:].drop(columns="_ts")

    logger.info(f"[split] train {len(train):,} ({train[time_col].min()[:10]} -> "
               f"{train[time_col].max()[:10]})")
    logger.info(f"        val   {len(val):,}")
    logger.info(f"        test  {len(test):,} ({test[time_col].min()[:10]} -> "
               f"{test[time_col].max()[:10]})")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        logger.info(f"        {name} fraud rate: {part[TARGET].mean()*100:.3f}%")
    return train, val, test


# ── the preparer ─────────────────────────────────────────────────────────────
class ModelPrep:
    """Fit on TRAIN, transform anything.

    Cardinality policy (thresholds are configurable):
        <= freq_threshold          -> category dtype only
        <= drop_original_above     -> category dtype PLUS a _FE frequency column
        >  drop_original_above     -> _FE column only; original dropped as the
                                      raw levels are too sparse to be useful
    """

    def __init__(self, freq_threshold: int = 20, drop_original_above: int = 200,
                 drop_zero_variance: bool = True):
        self.freq_threshold = freq_threshold
        self.drop_original_above = drop_original_above
        self.drop_zero_variance = drop_zero_variance
        self.feature_cols: list[str] = []
        self.freq_maps: dict[str, dict] = {}
        self.cat_levels: dict[str, pd.Index] = {}
        self.dropped: dict[str, str] = {}

    # -- fit ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        df = train_df.drop(columns=[c for c in METADATA_COLS if c in train_df.columns])

        for col in list(df.columns):
            series = df[col]

            if self.drop_zero_variance and series.nunique(dropna=True) <= 1:
                self.dropped[col] = "zero variance (constant or all-null)"
                df = df.drop(columns=col)
                continue

            if _is_bool_like(series) or pd.api.types.is_numeric_dtype(series):
                continue                                    # numeric passes through

            n_unique = series.nunique(dropna=True)
            if n_unique > self.freq_threshold:
                # rarity is the signal: map each value to its TRAIN frequency
                self.freq_maps[col] = (series.value_counts(normalize=True, dropna=True)
                                             .to_dict())
            if n_unique > self.drop_original_above:
                self.dropped[col] = f"cardinality {n_unique} > {self.drop_original_above}; kept as _FE only"
                df = df.drop(columns=col)
            else:
                self.cat_levels[col] = pd.Index(sorted(series.dropna().unique()))

        self.feature_cols = list(df.columns) + [f"{c}_FE" for c in self.freq_maps]
        self.feature_cols = sorted(set(self.feature_cols))

        logger.info(f"[prep] {len(self.feature_cols)} features "
                   f"({len(self.cat_levels)} categorical, {len(self.freq_maps)} frequency-encoded)")
        if self.dropped:
            logger.info(f"[prep] dropped {len(self.dropped)}:")
            for c, why in self.dropped.items():
                logger.info(f"          {c:44s} {why}")
        return self

    # -- transform ------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the feature matrix. Uses ONLY train-learned maps and levels."""
        if not self.feature_cols:
            raise RuntimeError("ModelPrep.transform called before fit().")

        out = pd.DataFrame(index=df.index)

        # frequency columns: unseen values -> -1, a distinct 'never seen in train' marker
        for col, vc in self.freq_maps.items():
            src = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
            out[f"{col}_FE"] = src.map(vc).fillna(-1).astype("float32")

        for col in self.feature_cols:
            if col.endswith("_FE") and col[:-3] in self.freq_maps:
                continue                                    # already built
            src = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
            if col in self.cat_levels:
                # pin TRAIN levels: values unseen in training become NaN rather than
                # silently introducing a new code
                out[col] = pd.Categorical(src, categories=self.cat_levels[col])
            elif src.dtype == bool:
                out[col] = src.astype("int8")
            else:
                out[col] = pd.to_numeric(src, errors="coerce")

        return out[self.feature_cols]

    # -- persistence ----------------------------------------------------------
    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"[prep] saved -> {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)


# ── orchestrator ─────────────────────────────────────────────────────────────
def prepare(training_set_path=None, artifacts_dir=None,
            test_frac=0.20, val_frac=0.15, **prep_kwargs):
    """training_set -> everything the training code needs.

    `training_set_path` defaults to config.TRAINING_SET_PATH, `artifacts_dir`
    to config.ARTIFACTS_DIR. Pass artifacts_dir=False (not just omit it) to
    skip saving the fitted ModelPrep.

    Returns a dict with X/y for each split, plus the realised amounts the cost
    model consumes. Amounts are returned SEPARATELY from features precisely so
    they cannot leak in."""
    setup_logging()
    training_set_path = training_set_path or config.TRAINING_SET_PATH
    if artifacts_dir is None:
        artifacts_dir = config.ARTIFACTS_DIR

    df = read_table(_resolve(training_set_path))
    logger.info(f"[prep] loaded {len(df):,} rows x {df.shape[1]} cols")

    if "label_source" in df.columns:
        # sanity: prove to yourself why this is excluded
        xt = pd.crosstab(df["label_source"], df[TARGET])
        perfect = ((xt > 0).sum(axis=1) == 1).all()
        logger.info(f"[prep] label_source maps 1:1 to label: {perfect} -> excluded from features")

    train, val, test = temporal_split(df, test_frac, val_frac)

    prep = ModelPrep(**prep_kwargs).fit(train)

    out = {"prep": prep}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out[f"X_{name}"] = prep.transform(part)
        out[f"y_{name}"] = part[TARGET].astype("int8").reset_index(drop=True)
        amt = part[AMOUNT_COL] if AMOUNT_COL in part.columns else part["send_amount"]
        out[f"amounts_{name}"] = pd.to_numeric(amt, errors="coerce").to_numpy()

    # schema assertions -- cheap, and they catch the expensive mistakes
    assert list(out["X_train"].columns) == list(out["X_test"].columns), "schema mismatch"
    for m in METADATA_COLS:
        assert m not in out["X_train"].columns, f"METADATA LEAK: {m} reached the features"
    logger.info(f"[prep] OK -- {out['X_train'].shape[1]} features, no metadata leaked")

    if artifacts_dir:
        prep.save(_resolve(artifacts_dir) / "model_prep.joblib")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-set", default=config.TRAINING_SET_PATH)
    ap.add_argument("--artifacts", default=config.ARTIFACTS_DIR)
    ap.add_argument("--freq-threshold", type=int, default=20)
    args = ap.parse_args()

    d = prepare(args.training_set, args.artifacts, freq_threshold=args.freq_threshold)
    logger.info(f"X_train {d['X_train'].shape}   X_val {d['X_val'].shape}   X_test {d['X_test'].shape}")