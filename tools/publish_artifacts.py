#!/usr/bin/env python
"""
publish_artifacts.py — package artifacts and publish them to a GitHub Release.

    python tools/publish_artifacts.py --tag model-v1

WHY A RELEASE RATHER THAN A COMMIT
----------------------------------
Release assets live OUTSIDE git history. The repo stays clean no matter how often
you retrain, and old artifacts do not accumulate in every clone forever. You get
a stable download URL that a deploy environment can fetch at startup.

(For this project the artifacts total ~200 KB, so committing them directly is
also perfectly reasonable. Use whichever fits your policy.)

WORKFLOW
    1. train locally            python pipeline/train.py --trials 20
    2. publish                  python tools/publish_artifacts.py --tag model-v1
    3. set on Render            ARTIFACT_URL=<printed URL>
    4. redeploy                 the service downloads the model at startup

Requires the GitHub CLI (`gh`) to be installed and authenticated, or pass
--no-upload to build the archive and upload it manually.
"""

import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only what the scoring service needs. optuna.db is a tuning journal, not a
# runtime dependency, and it grows with every study.
INCLUDE = ["model_prep.joblib", "calibrator.joblib", "final_xgb.json",
           "final_lgbm.joblib", "training_results.json", "cost_grid.json"]


def build_archive(artifacts_dir: Path, out: Path) -> Path:
    present = [f for f in INCLUDE if (artifacts_dir / f).exists()]
    if not present:
        sys.exit(f"No artifacts found in {artifacts_dir}. Train a model first.")

    models = [f for f in present if f.startswith("final_")]
    if not models:
        sys.exit("No model file (final_xgb.json / final_lgbm.joblib) present.")

    with tarfile.open(out, "w:gz") as t:
        for f in present:
            t.add(artifacts_dir / f, arcname=f)      # flat, no nested dir

    size_kb = out.stat().st_size / 1024
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"packaged {len(present)} files -> {out.name} ({size_kb:.0f} KB)")
    for f in present:
        print(f"    {f}")
    print(f"\nsha256: {digest}")
    return out


def upload(archive: Path, tag: str, notes: str) -> None:
    if not subprocess.run(["gh", "--version"], capture_output=True).returncode == 0:
        sys.exit("GitHub CLI (gh) not found. Install it, or use --no-upload and "
                 "attach the archive to a release manually.")

    exists = subprocess.run(["gh", "release", "view", tag],
                            capture_output=True).returncode == 0
    if exists:
        print(f"release {tag} exists; replacing the asset")
        subprocess.run(["gh", "release", "upload", tag, str(archive), "--clobber"],
                       check=True)
    else:
        subprocess.run(["gh", "release", "create", tag, str(archive),
                        "--title", f"Model artifacts {tag}", "--notes", notes],
                       check=True)

    repo = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner",
                           "-q", ".nameWithOwner"],
                          capture_output=True, text=True, check=True).stdout.strip()
    url = f"https://github.com/{repo}/releases/download/{tag}/{archive.name}"
    print(f"\nSet this on Render as an environment variable:\n\n    ARTIFACT_URL={url}\n")
    print("If the repo is PRIVATE, also set ARTIFACT_TOKEN to a PAT with repo scope.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="model-latest")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--out", default=str(ROOT / "artifacts.tgz"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--notes", default="Trained model artifacts for the scoring service.")
    a = ap.parse_args()

    archive = build_archive(Path(a.artifacts), Path(a.out))
    if a.no_upload:
        print(f"\nArchive ready: {archive}\nAttach it to a GitHub Release manually.")
    else:
        upload(archive, a.tag, a.notes)


if __name__ == "__main__":
    main()
