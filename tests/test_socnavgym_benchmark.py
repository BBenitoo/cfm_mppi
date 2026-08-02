import json
from pathlib import Path
import shutil
import tempfile
import unittest

from cfm_mppi.evaluation.socnavgym_benchmark import (
    BenchmarkContractError,
    DEFAULT_BENCHMARK_SUITE,
    load_benchmark_suite,
    sha256_file,
)


class SocNavGymBenchmarkManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_benchmark_suite()

    def test_locked_matrix_and_seed_schedule(self):
        suite = self.suite
        self.assertEqual(suite.suite_id, "socnavgym-v1-crowd-density-v1")
        self.assertEqual(suite.development_seeds, (17, 29))
        self.assertEqual(suite.benchmark_seeds, tuple(range(1000, 1030)))
        self.assertTrue(
            set(suite.development_seeds).isdisjoint(suite.benchmark_seeds)
        )
        self.assertEqual(len(suite.scenarios), 6)
        self.assertEqual(len(suite.jobs), 180)
        self.assertEqual(suite.expected_scenario_seed_pairs, 180)

        matrix = [
            (
                scenario.id,
                scenario.role,
                scenario.human_policy,
                scenario.dynamic_humans,
            )
            for scenario in suite.scenarios
        ]
        self.assertEqual(
            matrix,
            [
                ("orca_sparse_2", "primary", "orca", 2),
                ("orca_medium_5", "primary", "orca", 5),
                ("orca_dense_8", "primary", "orca", 8),
                ("sfm_sparse_2", "sensitivity", "sfm", 2),
                ("sfm_medium_5", "sensitivity", "sfm", 5),
                ("sfm_dense_8", "sensitivity", "sfm", 8),
            ],
        )
        self.assertTrue(
            all(scenario.map_size_metres == (10, 10) for scenario in suite.scenarios)
        )

    def test_job_indices_are_stable_and_scenario_major(self):
        expected = {
            0: ("orca_sparse_2", (1000,)),
            29: ("orca_sparse_2", (1029,)),
            30: ("orca_medium_5", (1000,)),
            179: ("sfm_dense_8", (1029,)),
        }
        for index, (scenario_id, seeds) in expected.items():
            job = self.suite.job(index)
            self.assertEqual(job.index, index)
            self.assertEqual(job.scenario.id, scenario_id)
            self.assertEqual(job.env_seeds, seeds)
            self.assertEqual(len(job.seed_indices), 1)

        with self.assertRaises(IndexError):
            self.suite.job(-1)
        with self.assertRaises(IndexError):
            self.suite.job(180)

    def test_planning_budget_is_frozen(self):
        self.assertEqual(
            dict(self.suite.planner),
            {
                "selection": "both",
                "horizon": 80,
                "max_history": 10,
                "cfm_candidates": 200,
                "branches": 10,
                "mppi_samples_per_branch": 200,
            },
        )
        self.assertEqual(self.suite.planner_seed_offset, 1_000_000)
        self.assertEqual(self.suite.seeds_per_job, 1)
        self.assertEqual(self.suite.device, "cuda")
        self.assertEqual(self.suite.cuda_device_contains, "H100")
        self.assertEqual(self.suite.dgl_backend, "pytorch")
        self.assertEqual(
            self.suite.rvo2_module_sha256,
            "07975350aae97e876636ae62e1564d5b76cd788b5945d17d4b3a711e3b99391d",
        )
        self.assertEqual(
            self.suite.human_goal_reached_policy, "geometric-only-v1"
        )
        self.assertEqual(
            self.suite.socnavgym_commit,
            "1ef13ee604b71730e9ec7f2d9fd9cb2e8b796549",
        )

    def test_declared_hash_rejects_config_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "benchmark_v1"
            shutil.copytree(DEFAULT_BENCHMARK_SUITE.parent, copied)
            config = copied / "orca_sparse_2.yaml"
            config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkContractError, "config hash mismatch"):
                load_benchmark_suite(copied / "suite.json")

    def test_seed_sets_must_remain_disjoint_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "benchmark_v1"
            shutil.copytree(DEFAULT_BENCHMARK_SUITE.parent, copied)
            seed_path = copied / "seeds.json"
            seed_document = json.loads(seed_path.read_text(encoding="utf-8"))
            seed_document["development_seeds"] = [1000]
            seed_path.write_text(
                json.dumps(seed_document, indent=2) + "\n", encoding="utf-8"
            )
            manifest_path = copied / "suite.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["seed_file_sha256"] = sha256_file(seed_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(BenchmarkContractError, "must be disjoint"):
                load_benchmark_suite(manifest_path)


if __name__ == "__main__":
    unittest.main()
