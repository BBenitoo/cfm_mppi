import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUN_REAL_CONTRACT = os.environ.get("RUN_SOCNAVGYM_CONTRACT") == "1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    RUN_REAL_CONTRACT,
    "set RUN_SOCNAVGYM_CONTRACT=1 in the configured HPC environment",
)
class SocNavGymContractIntegrationTest(unittest.TestCase):
    def test_pinned_world_frame_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "socnavgym-probe.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/probe_socnavgym_hpc.py",
                    "--candidate",
                    "v1-1ef13ee",
                    "--steps",
                    "32",
                    "--determinism-steps",
                    "8",
                    "--timeout-seconds",
                    "180",
                    "--output",
                    str(report_path),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=200,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                "SocNavGym contract probe failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
