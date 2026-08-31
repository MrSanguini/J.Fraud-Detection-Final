"""
run_pipeline.py: run the whole pipeline in order.

    python src/run_pipeline.py                       # full run
    python src/run_pipeline.py --chargebacks FILE    # include partner sheet
    python src/run_pipeline.py --skip-profile        # skip schema profiling
    python src/run_pipeline.py --limit 1000          # cap each source collection to N docs
    python src/run_pipeline.py --batch-size 50000    # extractor chunk size (RAM/speed tradeoff)
    python src/run_pipeline.py --force               # re-run every stage, ignore checkpoints
    python src/run_pipeline.py --from-stage D        # resume at assembly (re-run D onward)

Stages:
    A  profile_schema      detect provider-driven schema variance
    B  extractor           raw JSON -> clean flat tables (allowlist + hashing)
    C0 chargeback_parser   partner sheet -> confirmed-fraud labels (optional)
    C  label_builder       flag history + chargebacks -> per-transaction labels
    D  assembler           join + features -> training table
    E  test_pipeline       safety assertions (PII / identity / leakage)

CONFIG: paths, collection names, batch size, velocity windows, and
FP_UNIT_COST all live in config.py, not in this file or any stage module.
Switching the source from local JSON exports to a live Mongo/Atlas
deployment is config.DATA_SOURCE = "atlas" plus MONGO_URI/MONGO_DB as
environment variables; nothing in pipeline/*.py needs to change.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

import config
from common import setup_logging, table_exists

logger = logging.getLogger(__name__)


def _describe_result(result) -> str:
    """Row-count summary for the per-stage log line, from whatever shape a
    stage's run() happens to return (a DataFrame, a {name: rows} dict, or
    nothing)."""
    if isinstance(result, dict):
        return ", ".join(f"{k}={v:,}" for k, v in result.items()) if result else "no output"
    if hasattr(result, "shape"):                          # a DataFrame
        return f"{len(result):,} rows"
    return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chargebacks", help="partner chargeback CSV/XLSX")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--data-dir", default=config.DATA_DIR,
                    help="local export directory (ignored when "
                         "config.DATA_SOURCE == 'atlas')")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap each source collection to N documents "
                         "(cheap smoke run, e.g. before a real AWS run)")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                    help="documents per chunk in the extractor, so peak RAM "
                         "stays flat regardless of collection size")
    ap.add_argument("--force", action="store_true",
                    help="re-run every stage even if its output already exists")
    ap.add_argument("--from-stage", choices=["A", "B", "C0", "C", "D", "E"],
                    help="skip everything before this stage; run it (and "
                         "everything after) unconditionally")
    args = ap.parse_args()

    log_path = setup_logging()
    logger.info(f"run_pipeline starting, log file: {log_path}")
    pipeline_t0 = time.monotonic()

    import profile_schema, extractor, label_builder, assembler, test_pipeline

    logger.info(f"data source: {config.DATA_SOURCE!r}"
               + ("" if config.DATA_SOURCE == "files" else f" (db={config.MONGO_DB})"))
    extracted = ROOT / config.EXTRACTED_DIR

    stages = []
    if not args.skip_profile:
        stages.append(("A", "A  schema profile",
                       lambda: profile_schema.run(args.data_dir, limit=args.limit),
                       None))                            # no persisted output to checkpoint
    stages.append(("B", "B  extract",
                   lambda: extractor.run(args.data_dir, limit=args.limit,
                                         batch_size=args.batch_size),
                   extracted / "transfer"))
    if args.chargebacks:
        import chargeback_parser
        stages.append(("C0", "C0 chargebacks",
                       lambda: chargeback_parser.run(args.chargebacks),
                       extracted / "chargebacks"))
    stages.append(("C", "C  labels", label_builder.run, extracted / "labels"))
    stages.append(("D", "D  assemble", assembler.run, ROOT / config.TRAINING_SET_PATH))

    force_all = False
    if args.from_stage:
        codes = [s[0] for s in stages]
        if args.from_stage == "E":
            idx = len(stages)                             # skip straight to the safety check
        elif args.from_stage in codes:
            idx = codes.index(args.from_stage)
        else:
            logger.error(f"--from-stage {args.from_stage!r} isn't in this run's plan "
                        f"({', '.join(codes) or '(none)'}, E); check --skip-profile / "
                        f"--chargebacks match what you intend to resume.")
            sys.exit(1)
        skipped = stages[:idx]
        stages = stages[idx:]
        if skipped:
            logger.info(f"[resume] --from-stage {args.from_stage}: excluding "
                       f"{', '.join(s[0] for s in skipped)} from this run.")
        force_all = True                                  # stages from here on always execute

    for _code, name, fn, marker in stages:
        logger.info(f"===== STAGE {name} =====")
        if marker is not None and not args.force and not force_all and table_exists(marker):
            logger.info(f"[skip] {name}: output already exists at {marker} "
                       f"(use --force to re-run)")
            continue
        t0 = time.monotonic()
        try:
            result = fn()
        except Exception:
            logger.exception(f"STAGE FAILED: {name}")
            sys.exit(1)
        elapsed = time.monotonic() - t0
        logger.info(f"[timing] {name}: {elapsed:.2f}s, {_describe_result(result)}")

    logger.info("===== STAGE E  safety check =====")
    t0 = time.monotonic()
    passed = test_pipeline.run_safety_tests()
    elapsed = time.monotonic() - t0
    logger.info(f"[timing] E  safety check: {elapsed:.2f}s")
    if not passed:
        logger.error("SAFETY CHECK FAILED: do NOT use this training set.")
        sys.exit(1)

    total_elapsed = time.monotonic() - pipeline_t0
    logger.info(f"Pipeline complete in {total_elapsed:.2f}s. Log: {log_path}")


if __name__ == "__main__":
    main()
