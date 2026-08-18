"""
run_pipeline.py — run the whole pipeline in order.

    python src/run_pipeline.py                       # full run
    python src/run_pipeline.py --chargebacks FILE    # include partner sheet
    python src/run_pipeline.py --skip-profile        # skip schema profiling

Stages:
    A  profile_schema      detect provider-driven schema variance
    B  extractor           raw JSON -> clean flat tables (allowlist + hashing)
    C0 chargeback_parser   partner sheet -> confirmed-fraud labels (optional)
    C  label_builder       flag history + chargebacks -> per-transaction labels
    D  assembler           join + features -> training table
    E  test_pipeline       safety assertions (PII / identity / leakage)

Stage E is not optional in spirit: it is the check that this ran correctly on
data nobody watched it process.
"""

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chargebacks", help="partner chargeback CSV/XLSX")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    import profile_schema, extractor, label_builder, assembler, test_pipeline

    steps = []
    if not args.skip_profile:
        steps.append(("A  schema profile", lambda: profile_schema.run(args.data_dir)))
    steps.append(("B  extract", lambda: extractor.run(args.data_dir)))
    if args.chargebacks:
        import chargeback_parser
        steps.append(("C0 chargebacks", lambda: chargeback_parser.run(args.chargebacks)))
    steps.append(("C  labels", label_builder.run))
    steps.append(("D  assemble", assembler.run))

    for name, fn in steps:
        print(f"\n{'#'*70}\n# {name}\n{'#'*70}")
        try:
            fn()
        except Exception:
            print(f"\n!! STAGE FAILED: {name}\n")
            traceback.print_exc()
            sys.exit(1)

    print(f"\n{'#'*70}\n# E  safety check\n{'#'*70}")
    if not test_pipeline.check_training_set():
        print("\n!! SAFETY CHECK FAILED — do NOT use this training set.\n")
        sys.exit(1)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
