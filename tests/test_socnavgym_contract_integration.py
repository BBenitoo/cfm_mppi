import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from cfm_mppi.evaluation.socnavgym_adapter import (
    HUMAN_GOAL_REACHED_POLICY,
    SocNavGymAdapter,
)


RUN_REAL_CONTRACT = os.environ.get("RUN_SOCNAVGYM_CONTRACT") == "1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    RUN_REAL_CONTRACT,
    "set RUN_SOCNAVGYM_CONTRACT=1 in the configured HPC environment",
)
class SocNavGymContractIntegrationTest(unittest.TestCase):
    def test_adapter_installs_wall_clock_independent_human_goal_policy(self) -> None:
        config = (
            REPOSITORY_ROOT
            / "configs"
            / "socnavgym"
            / "benchmark_v1"
            / "orca_sparse_2.yaml"
        )
        with SocNavGymAdapter(config) as adapter:
            adapter.reset(seed=1000)
            human = adapter.unwrapped.dynamic_humans[0]
            human.initial_time = 0.0
            human.goal_x = human.x + 100.0
            human.goal_y = human.y
            offset = human.width / 2
            expected = np.hypot(
                human.x - human.goal_x, human.y - human.goal_y
            ) < (offset + human.goal_radius)
            self.assertEqual(human.has_reached_goal(), bool(expected))
            self.assertFalse(human.has_reached_goal())
            self.assertEqual(
                human.has_reached_goal.__func__.__cfm_mppi_goal_policy__,
                HUMAN_GOAL_REACHED_POLICY,
            )

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
