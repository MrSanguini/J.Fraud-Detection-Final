"""
Scoring API: HTTP wrapper around model/inference.py.

    uvicorn api.scoring.main:app --port 8001        (from the repo root)

Like the inference module it wraps, this service returns RISK, never a
decision. No approve/decline/block field, by design: banding and
decisioning belong to the consuming system.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# The ML code isn't an installed package, so put it on the path first.
# Kept self-contained (rather than shared with the training API) so this
# service can be deployed on its own.
ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (ROOT / "pipeline", ROOT / "model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import inference
from model_prep import ModelPrep  # noqa: F401  (needed to unpickle model_prep.joblib)

logger = logging.getLogger(__name__)

# Populated at startup by the lifespan handler below.
STATE: dict[str, Any] = {}


def _load_artifacts() -> dict:
    "Load the trained model, its calibrator, and the fitted preprocessor."
    import json

    art = Path(config.ARTIFACTS_DIR)

    # In a deploy environment (Render, etc.) the repo carries no artifacts --
    # they are gitignored build outputs. Fetch them from ARTIFACT_URL if absent.
    # No-op when they are already on disk, so local runs are unaffected.
    try:
        from artifact_store import ensure_artifacts
        fetched = ensure_artifacts(art)
        print(f"[artifact_store] ensure_artifacts returned {fetched}", flush=True)
    except ImportError as e:
        print(f"[artifact_store] IMPORT FAILED: {e}", flush=True)

    prep_path = art / "model_prep.joblib"
    cal_path = art / "calibrator.joblib"
    results_path = art / "training_results.json"

    missing = [p.name for p in (prep_path, cal_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing artifact(s) {missing} in {art}. Train a model first "
            f"(POST /train on the training API, or python pipeline/train.py)."
        )

    winner = None
    if results_path.exists():
        winner = json.loads(results_path.read_text()).get("winner")

    xgb_path, lgbm_path = art / "final_xgb.json", art / "final_lgbm.joblib"
    booster = None
    if winner == "lightgbm" or (winner is None and not xgb_path.exists()):
        if not lgbm_path.exists():
            raise FileNotFoundError(f"no model file found in {art}")
        model = joblib.load(lgbm_path)
    else:
        if not xgb_path.exists():
            raise FileNotFoundError(f"no model file found in {art}")
        import xgboost as xgb
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(str(xgb_path))
        # Only XGBoost supports the exact-SHAP path used for explanations;
        # inference.score() degrades to no explanation when booster is None.
        booster = model.get_booster()

    prep = joblib.load(prep_path)
    logger.info(f"loaded {winner or 'model'} with {len(prep.feature_cols)} features from {art}")
    return {"model": model, "calibrator": joblib.load(cal_path), "prep": prep,
            "booster": booster, "winner": winner or "unknown",
            "feature_order": list(prep.feature_cols)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts ONCE at startup, not per request: deserialising the
    model and preprocessor takes far longer than scoring with them."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        STATE.update(_load_artifacts())
    except Exception as e:
        # Start anyway so /health can report WHY it is unusable. Crashing at
        # boot in a container just yields a restart loop with the reason
        # buried in logs.
        logger.error(f"could not load artifacts: {e}")
        STATE["load_error"] = str(e)
    yield
    STATE.clear()


app = FastAPI(
    title="Fraud Scoring API",
    description="Calibrated fraud probability and its drivers. Returns risk, not a decision.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── request / response models ────────────────────────────────────────────────
# Below this share of expected features, a score is not trustworthy: the
# model is mostly reading nulls. Measured, not guessed -- a transaction
# carrying only send_amount (1/68) currently scores 0.85 "High", while the
# same full, legitimate transaction scores 0.00 "Low". Reported as a warning
# rather than a hard rejection so partial scoring stays possible; see the
# note in api/README.md about turning it into a 422.
MIN_FEATURE_COVERAGE = 0.5


class ScoreRequest(BaseModel):
    """`transaction` is a free-form dict rather than an enumerated schema on
    purpose: the authoritative field list is pipeline/schema_contract.py, and
    duplicating ~70 field definitions here would guarantee the two drift
    apart. Unknown keys are ignored; absent features become null."""
    transaction: dict[str, Any] = Field(..., description="One transaction, training_set-shaped")
    explain: bool = True
    top_k: int = Field(3, ge=1, le=20)


class BatchScoreRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(..., min_length=1)


def _require_model():
    if "model" not in STATE:
        raise HTTPException(status_code=503,
                            detail=f"model not loaded: {STATE.get('load_error', 'unknown error')}")


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    """JSON rows -> the DataFrame ModelPrep.transform() expects.
    ModelPrep.transform() tolerates missing columns (they become NaN, which
    the tree models handle natively)"""
    return pd.DataFrame(rows)


@app.get("/health")
def health():
    """Liveness plus whether the model is actually usable."""
    ready = "model" in STATE
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "model": STATE.get("winner"),
        "n_features": len(STATE.get("feature_order", [])),
        "artifacts_dir": str(config.ARTIFACTS_DIR),
        "error": STATE.get("load_error"),
    }


@app.post("/score")
def score(req: ScoreRequest):
    """Score ONE transaction, with the factors that drove the score."""
    _require_model()
    df = _to_frame([req.transaction])
    try:
        X = STATE["prep"].transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not prepare features: {e}")

    result = inference.score(
        STATE["model"], STATE["calibrator"], X.iloc[[0]],
        txn=req.transaction,
        booster=STATE["booster"] if req.explain else None,
        feature_order=STATE["feature_order"] if req.explain else None,
        k=req.top_k, explain_factors=req.explain,
    )

    expected = STATE["feature_order"]
    provided = sum(1 for f in expected if f in df.columns and df[f].notna().iloc[0])
    coverage = provided / len(expected) if expected else 0.0

    out = {**result.to_dict(),
           "features_provided": provided,
           "features_expected": len(expected)}
    if coverage < MIN_FEATURE_COVERAGE:
        # Loud, because the usual cause is a caller field-mapping bug, and the
        # symptom (a plausible-looking probability) is otherwise invisible.
        msg = (f"only {provided}/{len(expected)} features supplied "
               f"({coverage:.0%}); this score is computed mostly from missing "
               f"values and should not be trusted. Check the field names match "
               f"the pipeline's training_set columns.")
        logger.warning(msg)
        out["warning"] = msg
    return out


@app.post("/score/batch")
def score_batch(req: BatchScoreRequest):
    """Calibrated probabilities for many transactions, no explanations.

    Per-row SHAP is far too slow for bulk scoring; explain individual cases
    on demand via /score."""
    _require_model()
    df = _to_frame(req.transactions)
    try:
        X = STATE["prep"].transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not prepare features: {e}")

    probs = inference.score_batch(STATE["model"], STATE["calibrator"], X)
    return {"count": len(probs),
            "results": [{"probability": float(p),
                         "risk_band": inference.ScoreResult(probability=float(p)).risk_band}
                        for p in probs]}
