from pathlib import Path
import unittest

from cfm_mppi.evaluation.socnavgym_benchmark import load_benchmark_suite


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_V2_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v2" / "suite.json"
)


class SocNavGymBenchmarkV2ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_benchmark_suite(BENCHMARK_V2_SUITE)

    def test_locked_matrix_and_seed_schedule(self):
        suite = self.suite
        self.assertEqual(suite.suite_id, "socnavgym-v1-crowd-density-v2")
        self.assertEqual(suite.development_seeds, (17, 29))
        self.assertEqual(suite.benchmark_seeds, tuple(range(1000, 1050)))
        self.assertEqual(len(suite.scenarios), 6)
        self.assertEqual(len(suite.jobs), 300)
        self.assertEqual(suite.expected_scenario_seed_pairs, 300)
        self.assertEqual(
            [
                (
                    scenario.id,
                    scenario.role,
                    scenario.human_policy,
                    scenario.dynamic_humans,
                )
                for scenario in suite.scenarios
            ],
            [
                ("orca_sparse_5", "primary", "orca", 5),
                ("orca_medium_10", "primary", "orca", 10),
                ("orca_dense_15", "primary", "orca", 15),
                ("sfm_sparse_5", "sensitivity", "sfm", 5),
                ("sfm_medium_10", "sensitivity", "sfm", 10),
                ("sfm_dense_15", "sensitivity", "sfm", 15),
            ],
        )

    def test_job_mapping_and_execution_order_are_balanced(self):
        suite = self.suite
        boundaries = {
            0: ("orca_sparse_5", 1000),
            49: ("orca_sparse_5", 1049),
            50: ("orca_medium_10", 1000),
            149: ("orca_dense_15", 1049),
            150: ("sfm_sparse_5", 1000),
            299: ("sfm_dense_15", 1049),
        }
        for index, (scenario_id, seed) in boundaries.items():
            job = suite.job(index)
            self.assertEqual(job.scenario.id, scenario_id)
            self.assertEqual(job.env_seeds, (seed,))

        for scenario in suite.scenarios:
            jobs = [job for job in suite.jobs if job.scenario == scenario]
            self.assertEqual(sum(job.index % 2 == 0 for job in jobs), 25)
            self.assertEqual(sum(job.index % 2 == 1 for job in jobs), 25)

    def test_unchanged_planner_and_environment_contract(self):
        suite = self.suite
        self.assertEqual(
            dict(suite.planner),
            {
                "selection": "both",
                "horizon": 80,
                "max_history": 10,
                "cfm_candidates": 200,
                "branches": 10,
                "mppi_samples_per_branch": 200,
            },
        )
        self.assertEqual(suite.planner_seed_offset, 1_000_000)
        self.assertEqual(suite.seeds_per_job, 1)
        self.assertEqual(suite.output_subdirectory.as_posix(), "socnavgym/benchmark_v2")
        self.assertTrue(
            all(scenario.map_size_metres == (10, 10) for scenario in suite.scenarios)
        )


if __name__ == "__main__":
    unittest.main()
