#!/usr/bin/env python3
"""CLI entry point for the strict SocNavGym HPC contract probe."""

from pathlib import Path
import sys


# Running ``python scripts/...`` places only the scripts directory on sys.path.
# Keep the probe usable before the repository itself is installed on the HPC.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cfm_mppi.diagnostics.socnavgym_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
