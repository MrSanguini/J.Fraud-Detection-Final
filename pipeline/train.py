"""
train.py, L2.3: train, tune, calibrate, and evaluate the fraud model.

    python src/train.py --trials 20

DISCIPLINE THIS FILE ENFORCES
----------------------------
1. TEMPORAL SPLIT, never random. Handled upstream by model_prep.
2. EVERY DECISION IS MADE ON VALIDATION. Hyperparameters, the winning model, the
   calibrator, and the decision threshold are all chosen on val. Test is read
   ONCE, at the end, at the already-fixed threshold.
   Choosing the threshold on test means minimising cost against the very data you
   then report, which is optimistically biased. The gap between the val figure and
   the test figure IS that optimism -- expect test to look slightly worse, and
   trust it.
3. CALIBRATION IS NOT COSMETIC. scale_pos_weight inflates raw scores, so an
   uncalibrated "0.6" is not a 60% fraud probability. The rules layer thresholds
   on these probabilities and the UI shows them to humans, so they must mean what
   they say. Isotonic regression is monotonic, so it will NOT change AUROC/PR-AUC
   -- it fixes interpretability and downstream behaviour, not ranking.

SPEED NOTES (learned the hard way: a 30-trial run once took 10 hours)
   * tune on a ROW SUBSAMPLE; hyperparameters transfer, and it is ~3x faster
   * cap max_depth at 8; depth 9-10 trials are individually brutal
   * lower n_estimators during search, raise it for the final refit
   * PERSIST the Optuna study, so an interrupted run resumes instead of restarting
"""

import argparse
import json
import logging
import time
from pathlib import Path as _P

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config
import metrics
import model_prep
from common import setup_logging

logger = logging.getLogger(__name__)
ROOT = _P(__file__).resolve().parent.parent

SEED = config.SEED
FP_UNIT_COST = config.FP_UNIT_COST
ARTIFACTS_DIR = config.ARTIFACTS_DIR


def _resolve(p):
    p = _P(p)
    return p if p.is_absolute() else ROOT / p


# ── imbalance ────────────────────────────────────────────────────────────────
def scale_pos_weight(y) -> float:
    """neg/pos ratio: XGBoost's built-in lever for class imbalance.

    Derive it from the FIT data only -- computing it over val or test is a subtle
    leak of the label distribution you are about to be evaluated on."""
    y = np.asarray(y)
    pos = int(y.sum())
    neg = len(y) - pos
    spw = neg / max(pos, 1)
    logger.info(f"scale_pos_weight = {spw:.1f}  (neg {neg:,} / pos {pos:,})")
    return spw


# ── XGBoost version shims ────────────────────────────────────────────────────
# early_stopping_rounds moved between the constructor and .fit() across major
# versions. Try the modern form, fall back to the legacy one.
def _make_xgb(params: dict, spw: float, n_estimators: int, early_stop: int = 50):
    from xgboost import XGBClassifier
    kwargs = dict(
        n_estimators=n_estimators, **params,
        tree_method="hist",
        enable_categorical=True,          # consume pandas category dtypes directly
        scale_pos_weight=spw,
        eval_metric="aucpr",              # optimise the metric that matters at 2% fraud
        random_state=SEED, n_jobs=-1,
    )
    try:
        return XGBClassifier(early_stopping_rounds=early_stop, **kwargs)
    except TypeError:
        return XGBClassifier(**kwargs)


def _fit_xgb(model, X_fit, y_fit, X_val, y_val, early_stop: int = 50):
    try:
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=early_stop, verbose=False)
    return model


# ── tuning ───────────────────────────────────────────────────────────────────
def tune_xgb(X_fit, y_fit, X_val, y_val, spw, n_trials=20,
             subsample_frac=0.35, study_name="xgb_fraud", storage=None):
    """Optuna search on validation PR-AUC.

    Each trial fits on a SUBSAMPLE of the fit set (hyperparameters transfer well
    and it is roughly 3x faster) but always evaluates on the FULL validation set,
    so the selection signal stays honest.

    The study is persisted to SQLite: if the run dies at trial 15, re-running
    resumes at 16 rather than starting over."""
    import gc
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),          # capped: 9-10 are brutal
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        )
        sub = X_fit.sample(frac=subsample_frac, random_state=SEED)
        model = _make_xgb(params, spw, n_estimators=800)
        _fit_xgb(model, sub, y_fit.loc[sub.index], X_val, y_val)
        pr = metrics.ranking_metrics(y_val, model.predict_proba(X_val)[:, 1])["PR_AUC"]
        del model
        gc.collect()
        return pr

    if storage is None:
        storage = f"sqlite:///{_resolve(ARTIFACTS_DIR) / 'optuna.db'}"
        _resolve(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=study_name, storage=storage, load_if_exists=True,
    )
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    if done:
        logger.info(f"resuming persisted study: {done} trials already complete")
    remaining = max(0, n_trials - done)
    if remaining:
        t0 = time.monotonic()
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        logger.info(f"tuning took {(time.monotonic()-t0)/60:.1f} min")

    logger.info(f"best val PR-AUC {study.best_value:.4f}")
    logger.info(f"best params {study.best_params}")
    return study.best_params


# ── models ───────────────────────────────────────────────────────────────────
def train_final_xgb(X_fit, y_fit, X_val, y_val, spw, params, n_estimators=1500):
    """Refit on the FULL fit set with the winning parameters.

    Subsampling belongs to the search, not here: once good hyperparameters are
    found, train on everything."""
    model = _make_xgb(params, spw, n_estimators=n_estimators)
    _fit_xgb(model, X_fit, y_fit, X_val, y_val)
    best_it = getattr(model, "best_iteration", None)
    logger.info(f"final XGB stopped at iteration {best_it}")
    return model


def train_lgbm(X_fit, y_fit, X_val, y_val, spw):
    """LightGBM challenger.

    A second algorithm on identical data tells you whether XGBoost is actually the
    right tool or merely the familiar one. Two gotchas encoded below: `subsample`
    does nothing without `subsample_freq`, and verbose=-1 silences the chatter."""
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=1500, learning_rate=0.05, num_leaves=64,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    try:
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    except Exception:                      # metric name varies across versions
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


# ── calibration ──────────────────────────────────────────────────────────────
def calibrate(model, X_val, y_val):
    """Fit isotonic regression on VALIDATION scores.

    Monotonic, so ranking metrics are unchanged by construction. What it fixes:
    scale_pos_weight pushes raw scores upward, so the cost-minimising threshold
    lands somewhere meaningless like 0.61. After calibration a score of 0.9 means
    roughly a 90% chance of fraud -- which matters because the rules layer bands
    on these values and the UI shows them to people."""
    raw = model.predict_proba(X_val)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(raw, y_val)
    return cal


def predict_calibrated(model, calibrator, X):
    return calibrator.predict(model.predict_proba(X)[:, 1])


def reliability_table(y_true, probs, n_bins=10) -> pd.DataFrame:
    """Predicted vs actual by quantile bin. Well-calibrated means they track."""
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins,
                                            strategy="quantile")
    return pd.DataFrame({"predicted": mean_pred, "actual": frac_pos})


# ── orchestration ────────────────────────────────────────────────────────────
def run(training_set=None, trials=20, artifacts_dir=None, fp_unit_cost=None,
        skip_lgbm=False):
    setup_logging()

    fp_unit_cost = FP_UNIT_COST if fp_unit_cost is None else fp_unit_cost
    art = _resolve(artifacts_dir or ARTIFACTS_DIR)
    art.mkdir(parents=True, exist_ok=True)

    ts = training_set or config.TRAINING_SET_PATH
    d = model_prep.prepare(ts, artifacts_dir=art)
    X_fit, y_fit = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    amt_val, amt_test = d["amounts_val"], d["amounts_test"]

    spw = scale_pos_weight(y_fit)

    # ---- tune, refit, challenge -------------------------------------------
    best_params = tune_xgb(X_fit, y_fit, X_val, y_val, spw, n_trials=trials)
    xgb_model = train_final_xgb(X_fit, y_fit, X_val, y_val, spw, best_params)

    candidates = {"xgboost": xgb_model}
    if not skip_lgbm:
        try:
            candidates["lightgbm"] = train_lgbm(X_fit, y_fit, X_val, y_val, spw)
        except Exception as e:
            logger.warning(f"LightGBM challenger skipped: {e}")

    # ---- select the winner ON VALIDATION -----------------------------------
    logger.info("--- model selection (validation PR-AUC) ---")
    scored = {}
    for name, m in candidates.items():
        pr = metrics.ranking_metrics(y_val, m.predict_proba(X_val)[:, 1])["PR_AUC"]
        scored[name] = pr
        logger.info(f"    {name:10s} val PR-AUC {pr:.4f}")
    winner_name = max(scored, key=scored.get)
    winner = candidates[winner_name]
    logger.info(f"winner: {winner_name}")

    # ---- calibrate on val, then choose the threshold on val ----------------
    calibrator = calibrate(winner, X_val, y_val)
    cal_val = predict_calibrated(winner, calibrator, X_val)
    t_star, _, _ = metrics.best_threshold(y_val, cal_val, amt_val, fp_unit_cost)
    logger.info(f"threshold chosen on VALIDATION: {t_star:.3f}")

    # ---- ONE final read of test at the already-fixed threshold -------------
    cal_test = predict_calibrated(winner, calibrator, X_test)
    result = metrics.evaluate(y_test, cal_test, amt_test, t_star,
                              fp_unit_cost, label=f"TEST ({winner_name}+calibrated)")

    logger.info("--- reliability (predicted vs actual) ---")
    for _, r in reliability_table(y_test, cal_test).iterrows():
        logger.info(f"    predicted ~{r['predicted']:.3f} -> actual {r['actual']:.3f}")

    # ---- persist -----------------------------------------------------------
    if winner_name == "xgboost":
        winner.save_model(str(art / "final_xgb.json"))
    else:
        joblib.dump(winner, art / "final_lgbm.joblib")
    joblib.dump(calibrator, art / "calibrator.joblib")

    summary = {"winner": winner_name, "val_pr_auc": scored[winner_name],
               "threshold": float(t_star), "fp_unit_cost": fp_unit_cost,
               "best_params": best_params, **result}
    (art / "training_results.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"artifacts written to {art}")

    grid = metrics.cost_grid(y_test, cal_test, amt_test)
    (art / "cost_grid.json").write_text(json.dumps(grid, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-set", default=None)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--fp-unit-cost", type=float, default=None)
    ap.add_argument("--skip-lgbm", action="store_true")
    a = ap.parse_args()
    run(a.training_set, a.trials, a.artifacts, a.fp_unit_cost, a.skip_lgbm)