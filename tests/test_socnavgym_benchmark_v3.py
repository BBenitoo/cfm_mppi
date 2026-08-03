from pathlib import Path
import unittest

from cfm_mppi.evaluation.socnavgym_benchmark import load_benchmark_suite


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_V2_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v2" / "suite.json"
)
BENCHMARK_V3_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v3" / "suite.json"
)


class SocNavGymBenchmarkV3ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2_suite = load_benchmark_suite(BENCHMARK_V2_SUITE)
        cls.suite = load_benchmark_suite(BENCHMARK_V3_SUITE)

    def test_locked_fixed_human_matrix_and_seed_schedule(self):
        suite = self.suite
        self.assertEqual(suite.suite_id, "socnavgym-v1-fixed-10-v3")
        self.assertEqual(suite.development_seeds, (17, 29))
        self.assertEqual(suite.benchmark_seeds, tuple(range(1000, 1300)))
        self.assertEqual(len(suite.scenarios), 2)
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
                ("orca_fixed_10", "primary", "orca", 10, 0),
                ("sfm_fixed_10", "sensitivity", "sfm", 10, 0),
            ],
        )

    def test_job_mapping_and_execution_order_are_balanced(self):
        boundaries = {
            0: ("orca_fixed_10", 1000),
            299: ("orca_fixed_10", 1299),
            300: ("sfm_fixed_10", 1000),
            599: ("sfm_fixed_10", 1299),
        }
        for index, (scenario_id, seed) in boundaries.items():
            job = self.suite.job(index)
            self.assertEqual(job.scenario.id, scenario_id)
            self.assertEqual(job.env_seeds, (seed,))

        for scenario in self.suite.scenarios:
            jobs = [job for job in self.suite.jobs if job.scenario == scenario]
            self.assertEqual(len(jobs), 300)
            self.assertEqual(sum(job.index % 2 == 0 for job in jobs), 150)
            self.assertEqual(sum(job.index % 2 == 1 for job in jobs), 150)

        with self.assertRaises(IndexError):
            self.suite.job(600)

    def test_v2_planner_and_environment_contract_is_unchanged(self):
        suite = self.suite
        v2_suite = self.v2_suite
        self.assertEqual(dict(suite.planner), dict(v2_suite.planner))
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
            self.assertEqual(getattr(suite, attribute), getattr(v2_suite, attribute))
        self.assertEqual(
            suite.output_subdirectory.as_posix(), "socnavgym/benchmark_v3"
        )
        self.assertTrue(
            all(scenario.map_size_metres == (10, 10) for scenario in suite.scenarios)
        )

        v2_ten_human_configs = {
            scenario.human_policy: scenario.config_path.read_bytes()
            for scenario in v2_suite.scenarios
            if scenario.dynamic_humans == 10
        }
        for scenario in suite.scenarios:
            self.assertEqual(
                scenario.config_path.read_bytes(),
                v2_ten_human_configs[scenario.human_policy],
            )


if __name__ == "__main__":
    unittest.main()
