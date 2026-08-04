from pathlib import Path
import unittest

from cfm_mppi.evaluation.socnavgym_benchmark import load_benchmark_suite


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_V2_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v2" / "suite.json"
)
BENCHMARK_V4_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v4" / "suite.json"
)
BENCHMARK_V5_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v5" / "suite.json"
)


class SocNavGymBenchmarkV5ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2_suite = load_benchmark_suite(BENCHMARK_V2_SUITE)
        cls.v4_suite = load_benchmark_suite(BENCHMARK_V4_SUITE)
        cls.suite = load_benchmark_suite(BENCHMARK_V5_SUITE)

    def test_locked_density_matrix_route_and_seed_schedule(self):
        suite = self.suite
        self.assertEqual(
            suite.suite_id, "socnavgym-v1-fixed-route-crowd-density-v5"
        )
        self.assertEqual(suite.development_seeds, (17, 29))
        self.assertEqual(suite.benchmark_seeds, tuple(range(1000, 1100)))
        self.assertEqual(len(suite.scenarios), 6)
        self.assertEqual(len(suite.jobs), 600)
        self.assertEqual(suite.expected_scenario_seed_pairs, 600)
        self.assertEqual(
            [
                (
                    scenario.id,
                    scenario.role,
                    scenario.human_policy,
                    scenario.dynamic_humans,
                    scenario.static_humans,
                )
                for scenario in suite.scenarios
            ],
            [
                ("orca_sparse_5", "primary", "orca", 5, 0),
                ("orca_medium_10", "primary", "orca", 10, 0),
                ("orca_dense_15", "primary", "orca", 15, 0),
                ("sfm_sparse_5", "sensitivity", "sfm", 5, 0),
                ("sfm_medium_10", "sensitivity", "sfm", 10, 0),
                ("sfm_dense_15", "sensitivity", "sfm", 15, 0),
            ],
        )
        for scenario in suite.scenarios:
            self.assertEqual(scenario.map_size_metres, (10, 10))
            self.assertEqual(scenario.robot_start, (-4.0, -4.0))
            self.assertEqual(scenario.robot_goal, (4.0, 4.0))

    def test_job_mapping_and_execution_order_are_balanced(self):
        boundaries = {
            0: ("orca_sparse_5", 1000),
            99: ("orca_sparse_5", 1099),
            100: ("orca_medium_10", 1000),
            199: ("orca_medium_10", 1099),
            200: ("orca_dense_15", 1000),
            299: ("orca_dense_15", 1099),
            300: ("sfm_sparse_5", 1000),
            399: ("sfm_sparse_5", 1099),
            400: ("sfm_medium_10", 1000),
            499: ("sfm_medium_10", 1099),
            500: ("sfm_dense_15", 1000),
            599: ("sfm_dense_15", 1099),
        }
        for index, (scenario_id, seed) in boundaries.items():
            job = self.suite.job(index)
            self.assertEqual(job.scenario.id, scenario_id)
            self.assertEqual(job.env_seeds, (seed,))

        for scenario in self.suite.scenarios:
            jobs = [job for job in self.suite.jobs if job.scenario == scenario]
            self.assertEqual(len(jobs), 100)
            self.assertEqual(sum(job.index % 2 == 0 for job in jobs), 50)
            self.assertEqual(sum(job.index % 2 == 1 for job in jobs), 50)

        with self.assertRaises(IndexError):
            self.suite.job(600)

    def test_v4_runtime_and_planner_contract_is_unchanged(self):
        self.assertEqual(dict(self.suite.planner), dict(self.v4_suite.planner))
        for attribute in (
            "environment_id",
            "observation_wrapper",
            "socnavgym_commit",
            "human_goal_reached_policy",
            "checkpoint_sha256",
            "rvo2_module_sha256",
            "dgl_backend",
            "device",
            "cuda_device_contains",
            "planner_seed_offset",
            "seeds_per_job",
        ):
            self.assertEqual(
                getattr(self.suite, attribute), getattr(self.v4_suite, attribute)
            )
        self.assertEqual(
            self.suite.output_subdirectory.as_posix(), "socnavgym/benchmark_v5"
        )

    def test_environment_configs_match_v2_except_for_fixed_route_manifest(self):
        v2_configs = {
            (scenario.human_policy, scenario.dynamic_humans):
                scenario.config_path.read_bytes()
            for scenario in self.v2_suite.scenarios
        }
        for scenario in self.suite.scenarios:
            self.assertEqual(
                scenario.config_path.read_bytes(),
                v2_configs[(scenario.human_policy, scenario.dynamic_humans)],
            )


if __name__ == "__main__":
    unittest.main()
