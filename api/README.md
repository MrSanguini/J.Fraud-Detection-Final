# API layer

Two independent FastAPI services wrapping the ML code. No
feature engineering, no model logic, no thresholds. These files only load artifacts, and call in.

```
api/
  scoring/main.py     serve predictions   (fast, read-only, many replicas)
  training/main.py    run training jobs   (slow, writes artifacts, one instance)
```

They are kept as separate apps, each with its own path bootstrap, so either
can be deployed or copied on its own. They share no code with each other —
only the ML modules underneath.

## Running

From the repo root, in two terminals:

```bash
uvicorn api.scoring.main:app  --port 8001 --reload
uvicorn api.training.main:app --port 8002 --reload
```

Interactive docs at `http://localhost:8001/docs` and `:8002/docs`.

Both need `FRAUD_SALT` set (same as the pipeline), and read/write under
`config.OUTPUT_ROOT` — so `PIPELINE_OUTPUT_ROOT` redirects them too.

## Scoring API (:8001)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`       | liveness + whether the model actually loaded |
| POST | `/score`        | one transaction → probability + drivers |
| POST | `/score/batch`  | many transactions → probabilities only |

```bash
curl -X POST localhost:8001/score -H 'content-type: application/json' -d '{
  "transaction": {"send_amount": 250.0, "sending_from": "US", "sending_to": "NG",
                  "payment_method": "card", "hour": 3},
  "explain": true
}'
```

`transaction` is a free-form object shaped like a **`data/training_set` row**
(not the raw Mongo export, not an already-prepped feature matrix). It is
free-form on purpose: `pipeline/schema_contract.py` is the authoritative
field list, and restating ~70 fields here would guarantee drift. Unknown keys
are ignored; missing ones become null.

The service returns **risk, never a decision** — no approve/decline field.
Banding and decisioning belong to the consuming system, matching the
boundary `model/inference.py` already draws.

### ⚠ Partial payloads score badly, and look fine doing it

`ModelPrep.transform()` tolerates missing columns (they become NaN, which the
tree models handle natively), so an incomplete transaction still returns a
confident-looking number. Measured on the current model:

| payload | features | probability | band |
|---|---|---|---|
| full, legitimate transaction | 61/68 | 0.00 | Low |
| `{"send_amount": 250.0}` only | 1/68 | **0.85** | **High** |

The usual cause is a caller field-mapping bug (camelCase vs snake_case, say),
and the symptom is a plausible number rather than an error. So every `/score`
response carries `features_provided` / `features_expected`, and adds a
`warning` field below `MIN_FEATURE_COVERAGE` (currently 0.5).

**This is a warning, not a rejection.** If this API ever fronts a live payment
flow, consider making low coverage a hard `422` instead — one line in
`main.py`. That's a product call, so it isn't made for you here.

## Training API (:8002)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`          | liveness + whether a run is in flight |
| POST | `/train`           | start a run → `202` + `job_id` |
| GET  | `/train/{job_id}`  | poll one job |
| GET  | `/train`           | list jobs, newest first |

```bash
curl -X POST localhost:8002/train -H 'content-type: application/json' \
     -d '{"trials": 20, "skip_lgbm": false}'
# {"job_id": "41cb58849e88", "status": "queued", "poll": "/train/41cb58849e88"}

curl localhost:8002/train/41cb58849e88
```

**Training is asynchronous** because it takes minutes to hours (`train.py`
records a 30-trial run once taking 10 hours) — far past any HTTP timeout.

**One run at a time.** Training writes the shared artifacts directory that
the scoring API reads; two concurrent runs would interleave writes and leave
a mismatched model/calibrator/preprocessor set behind. A second `POST /train`
returns `409` while one is in flight.

**Building the training set is not this service's job.** `POST /train`
returns `409` if `data/training_set` is absent — run
`python pipeline/run_pipeline.py` first. Data preparation and model training
are separate concerns and stay that way.