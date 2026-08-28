"""
conftest.py — shared fixtures.

The expensive fixtures are SESSION-scoped: synthetic data is generated and the
pipeline is run ONCE for the whole test session, not per test. Regenerating per
test would make the suite unusably slow and would not test anything extra.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
TOOLS = ROOT / "tools"
MODEL = ROOT / "model"

# make the project modules importable without installing the package
for p in (PIPELINE, TOOLS, MODEL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Fixed salt for tests. Deterministic on purpose: hash-stability assertions
# depend on it. NEVER reuse a test salt in production.
TEST_SALT = "pytest-fixed-salt-do-not-use-in-production"

# Every pipeline output (extracted/, training_set.*, artifacts/, logs/) is
# redirected here via PIPELINE_OUTPUT_ROOT (see config.py), so running this
# suite never touches the real project's data/, artifacts/, or logs/
# directories.
TEST_OUTPUTS = Path(__file__).resolve().parent / "test_outputs"

# Set at CONFTEST MODULE IMPORT TIME, deliberately not inside a fixture.
# config.py reads PIPELINE_OUTPUT_ROOT once, at its own import time, and
# that import can happen during pytest's COLLECTION phase -- e.g.
# test_metrics.py's module-level `import metrics` pulls in config.py --
# which runs before ANY fixture, autouse or not. A fixture would set this
# too late; only module-level code in a conftest.py (loaded before test
# modules are collected) is early enough.
os.environ["FRAUD_SALT"] = TEST_SALT
os.environ["PIPELINE_OUTPUT_ROOT"] = str(TEST_OUTPUTS)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    """Generate a small synthetic dataset once per session.

    2,000 transfers is enough to exercise every code path (all four provider
    variants, append-only flag histories, chargebacks, velocity windows) while
    keeping the suite fast."""
    out = tmp_path_factory.mktemp("synthetic")
    r = subprocess.run(
        [sys.executable, str(TOOLS / "synthetic_gen.py"),
         "--n-transfers", "2000", "--seed", "1234", "--out", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.fail(f"synthetic data generation failed:\nSTDOUT:\n{r.stdout[-3000:]}\n"
                    f"STDERR:\n{r.stderr[-3000:]}")
    return out


@pytest.fixture(scope="session")
def pipeline_output(synthetic_dir, tmp_path_factory) -> Path:
    """Run the full pipeline once against the synthetic data.

    Returns the directory holding training_set.* and extracted/ -- under
    TEST_OUTPUTS, not the real project data/ (see PIPELINE_OUTPUT_ROOT in
    _test_env above). --force is still required even though output is
    isolated: TEST_OUTPUTS persists ACROSS test sessions on disk (it isn't
    a tmp_path), so a checkpoint left by a previous run would otherwise be
    silently reused instead of regenerated from this session's synthetic
    data -- run_pipeline.py's checkpointing can't tell "already correct"
    from "leftover from an earlier session"."""
    env = {**os.environ, "FRAUD_SALT": TEST_SALT, "PIPELINE_OUTPUT_ROOT": str(TEST_OUTPUTS)}
    r = subprocess.run(
        [sys.executable, str(PIPELINE / "run_pipeline.py"),
         "--data-dir", str(synthetic_dir),
         "--chargebacks", str(synthetic_dir / "partner_chargebacks.csv"),
         "--skip-profile", "--force"],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    if r.returncode != 0:
        pytest.fail(f"pipeline failed:\nSTDOUT:\n{r.stdout[-3000:]}\n"
                    f"STDERR:\n{r.stderr[-3000:]}")
    return TEST_OUTPUTS / "data"


@pytest.fixture(scope="session")
def training_set(pipeline_output):
    """The assembled training table as a DataFrame."""
    from common import read_table
    return read_table(pipeline_output / "training_set")


@pytest.fixture(scope="session")
def prepared(training_set, tmp_path_factory):
    """model_prep output: X/y splits plus the amounts the cost model needs."""
    import model_prep
    # no explicit training_set_path: defaults to config.TRAINING_SET_PATH,
    # already redirected to TEST_OUTPUTS by PIPELINE_OUTPUT_ROOT.
    return model_prep.prepare(artifacts_dir=False)   # don't write a real artifacts/ dir either


@pytest.fixture
def tiny_case():
    """A hand-computable four-row case used across the metrics tests.

        idx0  fraud, scored high -> caught      (no cost)
        idx1  fraud, scored low  -> MISSED $500 (fn_cost 500)
        idx2  legit, scored high -> WRONG BLOCK (fp_cost = fp_unit_cost)
        idx3  legit, scored low  -> approved    (no cost)
    """
    return {
        "y": [1, 1, 0, 0],
        "scores": [0.9, 0.1, 0.8, 0.2],
        "amounts": [100, 500, 300, 50],
        "threshold": 0.5,
    }
