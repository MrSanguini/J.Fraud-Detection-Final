"""
artifact_store.py — fetch model artifacts at startup in deploy environments.

WHY THIS EXISTS
---------------
Render (and most PaaS) deploy from a git repo, but model artifacts are gitignored:
they are build outputs, they change on every retrain, and binary blobs in git
history are permanent. So the serving container starts with no model.

This closes that gap. On startup the scoring service calls ensure_artifacts();
if the artifacts are already on disk it does nothing, otherwise it downloads and
unpacks them from a URL.

SOURCE OPTIONS (any HTTPS URL works)
    GitHub Releases   free, unlimited bandwidth, stored OUTSIDE git history,
                      2 GB per asset. Recommended -- the repo stays clean and the
                      artifact lives beside the code that produced it.
    S3 presigned URL  if artifacts are already in S3
    Hugging Face Hub  free model hosting, public or private
    Any static host

NOTE ON SIZE: this project's artifacts total ~200 KB. At that size, committing
them directly to git is entirely reasonable and simpler than this module. This
exists for the cases where that is not wanted -- policy, retrain frequency, or
artifacts that grow large enough to matter.
"""

import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Files the scoring service cannot start without.
REQUIRED = ("model_prep.joblib", "calibrator.joblib")
# At least one model file must be present.
MODEL_ANY = ("final_xgb.json", "final_lgbm.joblib")


def artifacts_present(dest: Path, required=REQUIRED, model_any=MODEL_ANY) -> bool:
    """True only if every required file AND at least one model file exist."""
    dest = Path(dest)
    if not all((dest / f).exists() for f in required):
        return False
    return any((dest / f).exists() for f in model_any)


def _download(url: str, target: Path, timeout: int = 120) -> Path:
    logger.info(f"downloading artifacts from {url.split('?')[0]}")
    req = urllib.request.Request(url, headers={"User-Agent": "fraud-pipeline"})
    token = os.environ.get("ARTIFACT_TOKEN")
    if token:                       # private GitHub release / authenticated host
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(target, "wb") as f:
        shutil.copyfileobj(r, f)
    logger.info(f"downloaded {target.stat().st_size / 1024:.0f} KB")
    return target


def _unpack(archive: Path, dest: Path) -> None:
    """Extract, flattening any single top-level directory in the archive.

    Release assets are commonly built as `tar -czf a.tgz artifacts/`, which nests
    everything one level deeper than expected; flattening avoids an
    artifacts/artifacts/ surprise."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(archive) as t:
                t.extractall(tmp)

        entries = list(tmp.iterdir())
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        for item in src.iterdir():
            target = dest / item.name
            if target.exists():
                target.unlink() if target.is_file() else shutil.rmtree(target)
            shutil.move(str(item), str(target))


def ensure_artifacts(dest, url: str = None, checksum: str = None,
                     force: bool = False) -> bool:
    """Make artifacts available at `dest`. Returns True if usable afterwards.

    Idempotent: a no-op when artifacts are already present, so calling it on
    every startup is safe and costs nothing after the first boot.

    `url` defaults to the ARTIFACT_URL environment variable, which is how a
    deploy environment supplies it without hardcoding a release tag.
    """
    dest = Path(dest)
    if not force and artifacts_present(dest):
        logger.info(f"artifacts already present at {dest}")
        return True

    url = url or os.environ.get("ARTIFACT_URL")
    if not url:
        logger.warning(
            "no artifacts on disk and ARTIFACT_URL is unset. The scoring service "
            "cannot start without a model. Either set ARTIFACT_URL to a release "
            "asset, or commit the artifacts (they are small)."
        )
        return False

    try:
        suffix = ".zip" if url.split("?")[0].endswith(".zip") else ".tgz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            archive = Path(tmp.name)
        _download(url, archive)

        if checksum:
            got = hashlib.sha256(archive.read_bytes()).hexdigest()
            if got != checksum:
                raise ValueError(f"checksum mismatch: expected {checksum}, got {got}")
            logger.info("checksum verified")

        _unpack(archive, dest)
        archive.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"could not fetch artifacts: {e}")
        return False

    if not artifacts_present(dest):
        logger.error(f"archive unpacked but required files are missing in {dest}. "
                     f"found: {[p.name for p in dest.iterdir()]}")
        return False

    logger.info(f"artifacts ready at {dest}")
    return True
