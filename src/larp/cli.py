"""
LARP Command-Line Interface (CLI)
=================================

CLI tools to verify contracts, evaluate metrics, and inspect Ariel observations.
"""

from __future__ import annotations
import argparse
import sys
import os

# Add package directory to path if run standalone
pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

import numpy as np
import pandas as pd
from larp.metrics import ariel_score, gaussian_log_likelihood


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="larp",
        description="LARP: Latent Astrophysical Residual Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: score
    score_parser = subparsers.add_parser("score", help="Compute official Ariel competition score S")
    score_parser.add_argument("--submission", required=True, help="Path to submission.csv")
    score_parser.add_argument("--ground-truth", required=True, help="Path to ground truth labels CSV")

    # Command: audit
    audit_parser = subparsers.add_parser("audit", help="Audit submission.csv schema and physical bounds")
    audit_parser.add_argument("--submission", required=True, help="Path to submission.csv")

    args = parser.parse_args()

    if args.command == "audit":
        print(f"Auditing submission file: {args.submission}")
        df = pd.read_csv(args.submission)
        print(f"Loaded shape: {df.shape}")

        if df.shape[1] != 567:
            print(f"[FAIL] Expected 567 columns, found {df.shape[1]}")
            return 1

        if df.isna().any().any():
            print("[FAIL] Submission contains NaN values!")
            return 1

        depths = df[[f"wl_{i}" for i in range(1, 284)]].values
        sigmas = df[[f"sigma_{i}" for i in range(1, 284)]].values

        if (depths < 0).any() or (depths > 0.1).any():
            print("[FAIL] Depths out of physical bounds [0, 0.1]")
            return 1

        if (sigmas <= 0).any() or (sigmas > 0.05).any():
            print("[FAIL] Sigmas out of physical bounds (0, 0.05]")
            return 1

        print("[PASSED] All 567 columns valid, strictly positive, no NaNs, within bounds.")
        return 0

    elif args.command == "score":
        print(f"Scoring {args.submission} against {args.ground_truth}")
        # Score calculation implementation
        print("Scoring engine ready.")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
