"""
Training API: end-to-end. Production data in, scoreable model artifacts out.

    uvicorn api.training.main:app --port 8002       (from the repo root)

One POST runs both halves of the job:

    raw production data
        -> pipeline/run_pipeline.py   (extract, label, assemble, SAFETY CHECK)
        -> data/training_set
        -> pipeline/train.py          (tune, calibrate, pick threshold)
        -> artifacts/                 (ready for the scoring API to serve)

DELIBERATELY THIN. No ML and no data logic lives here: the pipeline stages
are run_pipeline.py's job and the modelling is train.py's. This file only
sequences them, tracks the job, and reports on it.

WHY THE PIPELINE RUNS AS A SUBPROCESS
-------------------------------------
run_pipeline.py is a CLI: argparse plus sys.exit(). Calling main() in-process
would swallow its exit codes and leave a half-imported module set behind.
A subprocess also means a pipeline crash cannot take down this service, and
it is the only way to vary config.py's settings per job, since config reads
its environment at IMPORT time.

WHY TRAINING IS A JOB, NOT A REQUEST
------------------------------------
The pipeline is minutes on real volumes and tuning is minutes to hours
(train.py's docstring records a 30-trial run once taking 10 hours). That is
past every HTTP, proxy, and load-balancer timeout in between, so POST /train
returns a job id immediately and the work continues on a background thread.

THE SAFETY CHECK IS A HARD GATE
-------------------------------
run_pipeline.py's Stage E asserts no PII, no identity columns, no leakage, no
metadata in the feature matrix. If it fails the pipeline exits non-zero and
this service will NOT proceed to training. Training on a training set that
failed those assertions is exactly the outcome the check exists to prevent.
"""

import logging
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The ML code isn't an installed package, so put it on the path first.
# Kept self-contained (rather than shared with the scoring API) so this
# service can be deployed on its own.
ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (ROOT / "pipeline", ROOT / "model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)

RUN_PIPELINE = ROOT / "pipeline" / "run_pipeline.py"

# How much pipeline console output to keep on a job record. Enough to
# diagnose a failure through the API without having to reach the log file.
LOG_TAIL_CHARS = 4000

# In-memory job registry.
#
# LIMITATION, stated plainly: this lives in the process. Jobs vanish on
# restart, and with more than one uvicorn worker each worker sees only its
# own jobs (so a poll can 404 against a worker that did not run the job).
# Fine for one training box, which is the actual deployment. Anything more
# needs a real store (Redis, a table) and a proper queue.
JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    yield
    JOBS.clear()


app = FastAPI(
    title="Fraud Model Training API",
    description="End-to-end: runs the data pipeline, then trains. Asynchronous.",
    version="2.0.0",
    lifespan=lifespan,
)


class TrainRequest(BaseModel):
    """Pipeline options mirror run_pipeline.py's flags; model options mirror
    train.py's. Defaults come from config.py so the API and the command line
    behave identically."""

    # ── pipeline half ────────────────────────────────────────────────────
    data_dir: str | None = Field(
        None, description=f"Source data directory (default {config.DATA_DIR})")
    chargebacks: str | None = Field(
        None, description="Partner chargeback CSV/XLSX. Omitting it means labels come from "
                          "flag documents only, which caps label coverage.")
    limit: int | None = Field(
        None, ge=1, description="Cap each source collection to N documents (cheap smoke run)")
    skip_profile: bool = Field(True, description="Skip Stage A schema profiling")
    force: bool = Field(
        True, description="Re-run every pipeline stage, ignoring checkpoints. Leave TRUE for a "
                          "real retrain: with it off, existing data/extracted tables are reused "
                          "and you silently train on stale data.")
    skip_pipeline: bool = Field(
        False, description="Train on the EXISTING training_set without rebuilding it. For "
                           "re-tuning a model on unchanged data.")

    # ── training half ────────────────────────────────────────────────────
    trials: int = Field(20, ge=1, le=500, description="Optuna trials; more = slower")
    fp_unit_cost: float | None = Field(
        None, description=f"$ per wrongly-declined customer (default {config.FP_UNIT_COST})")
    skip_lgbm: bool = Field(False, description="Skip the LightGBM challenger model")
    training_set: str | None = Field(
        None, description="Override the training_set path; defaults to config.TRAINING_SET_PATH")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secs(start: datetime) -> float:
    return round((datetime.now(timezone.utc) - start).total_seconds(), 1)


def _stage(job_id: str, name: str, **fields) -> None:
    JOBS[job_id].setdefault("stages", {}).setdefault(name, {}).update(**fields)


def _run_pipeline(job_id: str, req: TrainRequest) -> tuple[int, str]:
    """Run run_pipeline.py as a subprocess. Returns (returncode, output tail)."""
    cmd = [sys.executable, str(RUN_PIPELINE)]
    if req.data_dir:
        cmd += ["--data-dir", req.data_dir]
    if req.chargebacks:
        cmd += ["--chargebacks", req.chargebacks]
    if req.limit is not None:
        cmd += ["--limit", str(req.limit)]
    if req.skip_profile:
        cmd += ["--skip-profile"]
    if req.force:
        cmd += ["--force"]

    logger.info(f"[job {job_id}] pipeline: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    tail = ((r.stdout or "") + (r.stderr or ""))[-LOG_TAIL_CHARS:]
    return r.returncode, tail


def _run_job(job_id: str, req: TrainRequest) -> None:
    """Executed on a background thread: pipeline, then training.

    Every exit path must record a terminal state, or a caller polls a
    'running' job forever."""
    import train

    t0 = datetime.now(timezone.utc)
    JOBS[job_id].update(status="running", started_at=_now())

    try:
        # ── pipeline ─────────────────────────────────────────────────────
        if req.skip_pipeline:
            _stage(job_id, "pipeline", status="skipped",
                   detail="skip_pipeline=true; training on the existing training_set")
        else:
            JOBS[job_id]["stage"] = "pipeline"
            p0 = datetime.now(timezone.utc)
            _stage(job_id, "pipeline", status="running", started_at=_now())
            rc, tail = _run_pipeline(job_id, req)
            if rc != 0:
                # Includes a failed Stage E safety check. Do NOT train.
                _stage(job_id, "pipeline", status="failed", returncode=rc,
                       duration_s=_secs(p0), output_tail=tail)
                JOBS[job_id].update(
                    status="failed", stage="pipeline", finished_at=_now(),
                    error=f"pipeline exited {rc}; training was not started. "
                          f"If this was the Stage E safety check, the training set is "
                          f"unsafe to train on: see output_tail.")
                logger.error(f"[job {job_id}] pipeline failed (rc={rc}); not training")
                return
            _stage(job_id, "pipeline", status="succeeded", duration_s=_secs(p0),
                   output_tail=tail)
            logger.info(f"[job {job_id}] pipeline finished in {_secs(p0)}s")

        # ── training ─────────────────────────────────────────────────────
        JOBS[job_id]["stage"] = "training"
        m0 = datetime.now(timezone.utc)
        _stage(job_id, "training", status="running", started_at=_now())
        summary = train.run(training_set=req.training_set, trials=req.trials,
                            fp_unit_cost=req.fp_unit_cost, skip_lgbm=req.skip_lgbm)
        _stage(job_id, "training", status="succeeded", duration_s=_secs(m0))

        JOBS[job_id].update(
            status="succeeded", stage=None, finished_at=_now(),
            duration_s=_secs(t0), result=summary,
            artifacts_dir=str(config.ARTIFACTS_DIR),
            note="artifacts written. The scoring API caches the model at startup, "
                 "so restart it to serve this one.")
        logger.info(f"[job {job_id}] complete in {_secs(t0)}s")

    except Exception as e:
        logger.exception(f"[job {job_id}] failed")
        stage = JOBS[job_id].get("stage")
        if stage:
            _stage(job_id, stage, status="failed")
        JOBS[job_id].update(status="failed", finished_at=_now(),
                            error=f"{type(e).__name__}: {e}")
    finally:
        with _LOCK:
            JOBS[job_id]["_active"] = False


def _active_job() -> str | None:
    return next((jid for jid, j in JOBS.items() if j.get("_active")), None)


@app.get("/health")
def health():
    running = _active_job()
    return {
        "status": "ok",
        "training_in_progress": running is not None,
        "active_job": running,
        "jobs_tracked": len(JOBS),
        "data_dir": str(config.DATA_DIR),
        "training_set": str(config.TRAINING_SET_PATH),
        "artifacts_dir": str(config.ARTIFACTS_DIR),
    }


@app.post("/train", status_code=202)
def start_training(req: TrainRequest, background: BackgroundTasks):
    """Run the pipeline and then train. Returns immediately with a job id.

    End-to-end by default: raw data -> training_set -> model artifacts. Pass
    `skip_pipeline: true` to re-tune on the existing training set instead."""
    from common import table_exists

    # Fail fast on the obvious misconfigurations, so the caller gets a clear
    # 4xx now rather than a job that dies three minutes in.
    if req.skip_pipeline:
        ts = req.training_set or config.TRAINING_SET_PATH
        if not table_exists(ts):
            raise HTTPException(
                status_code=409,
                detail=f"skip_pipeline=true but no training set exists at {ts}. "
                       f"Run with skip_pipeline=false to build one.")
    else:
        src = Path(req.data_dir) if req.data_dir else Path(config.ROOT) / config.DATA_DIR
        if not src.is_absolute():
            src = Path(config.ROOT) / src
        if not src.is_dir():
            raise HTTPException(status_code=400,
                                detail=f"source data directory not found: {src}")

    with _LOCK:
        if (running := _active_job()) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"job {running} is already running; concurrent runs would corrupt "
                       f"the shared data/ and artifacts/ directories. Wait for it to finish.")
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"job_id": job_id, "status": "queued", "stage": None,
                        "created_at": _now(), "params": req.model_dump(),
                        "stages": {}, "_active": True}

    background.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued",
            "steps": ["training"] if req.skip_pipeline else ["pipeline", "training"],
            "poll": f"/train/{job_id}",
            "note": "asynchronous; poll the URL above for stage-by-stage status"}


@app.get("/train/{job_id}")
def job_status(job_id: str):
    if (job := JOBS.get(job_id)) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.get("/train")
def list_jobs():
    """Newest first. Stage detail is omitted; GET /train/{job_id} has it."""
    jobs = [{k: v for k, v in j.items() if not k.startswith("_") and k != "stages"}
            for j in JOBS.values()]
    return {"count": len(jobs), "jobs": sorted(jobs, key=lambda j: j["created_at"], reverse=True)}
