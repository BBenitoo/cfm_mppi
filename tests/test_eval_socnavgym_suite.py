from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from cfm_mppi.evaluation.eval_socnavgym_suite import (
    _RUNTIME_METADATA_FIELDS,
    _ensure_run_contract,
    _write_new_json_atomic,
    aggregate_suite,
)
from cfm_mppi.evaluation.socnavgym_adapter import HumanState, SocNavState
from cfm_mppi.evaluation.socnavgym_benchmark import (
    BenchmarkContractError,
    CANONICAL_PLANNERS,
    benchmark_metadata,
    load_benchmark_suite,
    validate_shard_document,
)
from cfm_mppi.evaluation.socnavgym_runner import (
    EpisodeContext,
    EpisodeResult,
    PlanningBudget,
    StepRecord,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_V4_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v4" / "suite.json"
)


def _fake_state(human_count, *, robot_x):
    humans = tuple(
        HumanState(
            id=index,
            position=np.asarray([2.0 + index, float(index)], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            radius=0.36,
            orientation=0.1 * index,
            gaze=index % 2 == 0,
        )
        for index in range(1, human_count + 1)
    )
    return SocNavState(
        robot_state=np.asarray([robot_x, 0.0, 0.0], dtype=np.float32),
        goal=np.asarray([1.0, 0.0], dtype=np.float32),
        robot_body_velocity=np.zeros(3, dtype=np.float32),
        robot_radius=0.25,
        goal_radius=0.35,
        humans=humans,
    )


def _fake_shard(suite, job):
    summaries = []
    episodes = []
    human_count = job.scenario.dynamic_humans + job.scenario.static_humans
    for env_seed in job.env_seeds:
        for planner_index, planner in enumerate(CANONICAL_PLANNERS):
            success = planner_index == 0
            collision = planner_index == 1
            final_info = {
                "SUCCESS": success,
                "COLLISION": collision,
                "COLLISION_HUMAN": collision,
                "COLLISION_OBJECT": False,
                "COLLISION_WALL": False,
                "OUT_OF_MAP": False,
                "TIMEOUT": False,
                "TIME_TO_REACH_GOAL": 1.0 if success else None,
                "PATH_LENGTH": 0.1,
                "MINIMUM_DISTANCE_TO_HUMAN": 1.0,
            }
            initial_state = _fake_state(human_count, robot_x=0.0)
            next_state = _fake_state(human_count, robot_x=0.1)
            context = EpisodeContext(
                env_seed=env_seed,
                planner_seed=env_seed + suite.planner_seed_offset,
                time_step=0.1,
                episode_length=256,
                max_human_speed=0.8,
                control_low=np.asarray([-1.0, -1.0], dtype=np.float32),
                control_high=np.asarray([1.0, 1.0], dtype=np.float32),
            )
            step = StepRecord(
                step_index=0,
                state=initial_state,
                next_state=next_state,
                requested_control=np.asarray([0.5, 0.0], dtype=np.float32),
                applied_control=np.asarray([0.5, 0.0], dtype=np.float32),
                normalized_action=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
                reward=float(planner_index),
                terminated=True,
                truncated=False,
                info=final_info,
                diagnostics={},
                decision_seconds=0.2 + planner_index,
                bookkeeping_seconds=0.02 + planner_index,
            )
            episode = EpisodeResult(
                planner_name=planner,
                budget=PlanningBudget(
                    cfm_candidates=suite.planner["cfm_candidates"],
                    refinement_rollouts=(
                        suite.planner["branches"]
                        * suite.planner["mppi_samples_per_branch"]
                    ),
                ),
                context=context,
                initial_state=initial_state,
                reset_info={},
                steps=(step,),
                runner_limit_reached=False,
            ).to_dict()
            episodes.append(episode)
            summaries.append(deepcopy(episode["summary"]))
    planner_config = {
        key: suite.planner[key]
        for key in (
            "horizon",
            "max_history",
            "cfm_candidates",
            "branches",
            "mppi_samples_per_branch",
        )
    }
    return {
        "schema_version": "cfm_mppi.socnavgym_evaluation.v1",
        "environment_seeds": list(job.env_seeds),
        "planners": list(CANONICAL_PLANNERS),
        "execution_orders": [
            (
                list(CANONICAL_PLANNERS)
                if (job.index + seed_index) % 2 == 0
                else list(reversed(CANONICAL_PLANNERS))
            )
            for seed_index in range(len(job.env_seeds))
        ],
        "episodes": episodes,
        "summaries": summaries,
        "metadata": {
            "random_model_smoke_only": False,
            "repository_dirty": False,
            "repository_commit": "test-commit",
            "checkpoint_sha256": suite.checkpoint_sha256,
            "environment_config_sha256": job.scenario.config_sha256,
            "planner_seed_offset": suite.planner_seed_offset,
            "execution_order_offset": job.index % 2,
            "planner_config": planner_config,
            "implementation_sha256": "test-implementation",
            "python": "3.11.test",
            "platform": "test-platform",
            "torch": "test",
            "torch_cuda_runtime": "test",
            "cudnn": 1,
            "gymnasium": "test",
            "socnavgym": "test",
            "socnavgym_commit": suite.socnavgym_commit,
            "socnavgym_human_goal_policy": suite.human_goal_reached_policy,
            "numpy": "test",
            "dgl": "test",
            "pyrvo2": "test",
            "rvo2_module": {
                "path": "/test/rvo2.so",
                "sha256": suite.rvo2_module_sha256,
            },
            "device": "cuda",
            "cuda_device": "H100 test",
        },
        "benchmark": benchmark_metadata(suite, job),
    }


def _fake_run_contract(suite):
    metadata = _fake_shard(suite, suite.job(0))["metadata"]
    return {
        "schema_version": "cfm_mppi.socnavgym_run_contract.v1",
        "suite_id": suite.suite_id,
        "manifest_sha256": suite.manifest_sha256,
        "seed_file_sha256": suite.seed_sha256,
        "scenario_config_sha256": {
            scenario.id: scenario.config_sha256 for scenario in suite.scenarios
        },
        "checkpoint_sha256": suite.checkpoint_sha256,
        "rvo2_module_sha256": suite.rvo2_module_sha256,
        "dgl_backend": suite.dgl_backend,
        "planner_seed_offset": suite.planner_seed_offset,
        "planner": dict(suite.planner),
        "device": suite.device,
        "cuda_device_contains": suite.cuda_device_contains,
        "runtime_metadata": {
            field: metadata.get(field) for field in _RUNTIME_METADATA_FIELDS
        },
    }


class SocNavGymSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_benchmark_suite()

    def test_accepts_a_complete_locked_shard(self):
        job = self.suite.job(0)
        validate_shard_document(_fake_shard(self.suite, job), self.suite, job)

    def test_v4_rejects_a_shard_without_its_fixed_robot_route(self):
        suite = load_benchmark_suite(BENCHMARK_V4_SUITE)
        job = suite.job(0)
        with self.assertRaisesRegex(BenchmarkContractError, "robot start"):
            validate_shard_document(_fake_shard(suite, job), suite, job)

    def test_rejects_smoke_and_execution_order_drift(self):
        job = self.suite.job(0)
        smoke = _fake_shard(self.suite, job)
        smoke["metadata"]["random_model_smoke_only"] = True
        with self.assertRaisesRegex(BenchmarkContractError, "smoke"):
            validate_shard_document(smoke, self.suite, job)

        wrong_order = _fake_shard(self.suite, job)
        wrong_order["execution_orders"] = [list(reversed(CANONICAL_PLANNERS))]
        with self.assertRaisesRegex(BenchmarkContractError, "execution order"):
            validate_shard_document(wrong_order, self.suite, job)

    def test_rejects_non_h100_output_and_tampered_top_level_summary(self):
        job = self.suite.job(0)
        cpu = _fake_shard(self.suite, job)
        cpu["metadata"]["device"] = "cpu"
        cpu["metadata"]["cuda_device"] = None
        with self.assertRaisesRegex(BenchmarkContractError, "CUDA"):
            validate_shard_document(cpu, self.suite, job)

        tampered = _fake_shard(self.suite, job)
        tampered["summaries"][0]["return"] = 999999.0
        with self.assertRaisesRegex(BenchmarkContractError, "inconsistent"):
            validate_shard_document(tampered, self.suite, job)

    def test_rejects_missing_metric_and_zero_step_episode(self):
        job = self.suite.job(0)
        missing = _fake_shard(self.suite, job)
        del missing["summaries"][0]["minimum_human_clearance"]
        del missing["episodes"][0]["summary"]["minimum_human_clearance"]
        with self.assertRaisesRegex(BenchmarkContractError, "summary schema"):
            validate_shard_document(missing, self.suite, job)

        empty = _fake_shard(self.suite, job)
        empty["summaries"][0]["steps"] = 0
        empty["episodes"][0]["summary"]["steps"] = 0
        empty["episodes"][0]["steps"] = []
        with self.assertRaisesRegex(BenchmarkContractError, "at least one step"):
            validate_shard_document(empty, self.suite, job)

    def test_rejects_metrics_that_disagree_with_step_records(self):
        job = self.suite.job(0)
        mutations = {
            "return": 999999.0,
            "geometric_path_length": 999999.0,
            "minimum_human_clearance": 999999.0,
            "cold_start_decision_seconds": 999999.0,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                tampered = _fake_shard(self.suite, job)
                tampered["summaries"][0][field] = value
                tampered["episodes"][0]["summary"][field] = value
                with self.assertRaisesRegex(BenchmarkContractError, "step-derived"):
                    validate_shard_document(tampered, self.suite, job)

        tampered_latency = _fake_shard(self.suite, job)
        for summary in (
            tampered_latency["summaries"][0],
            tampered_latency["episodes"][0]["summary"],
        ):
            summary["decision_latency"]["mean"] = 999999.0
        with self.assertRaisesRegex(BenchmarkContractError, "step-derived"):
            validate_shard_document(tampered_latency, self.suite, job)

    def test_rejects_invalid_step_schema_and_negative_latency(self):
        job = self.suite.job(0)
        missing = _fake_shard(self.suite, job)
        del missing["episodes"][0]["steps"][0]["reward"]
        with self.assertRaisesRegex(BenchmarkContractError, "step schema"):
            validate_shard_document(missing, self.suite, job)

        negative = _fake_shard(self.suite, job)
        negative["episodes"][0]["steps"][0]["decision_seconds"] = -1.0
        with self.assertRaisesRegex(BenchmarkContractError, "non-negative"):
            validate_shard_document(negative, self.suite, job)

    def test_immutable_writer_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.json"
            _write_new_json_atomic(path, {"first": True})
            original = path.read_bytes()
            with self.assertRaisesRegex(BenchmarkContractError, "overwrite"):
                _write_new_json_atomic(path, {"second": True})
            self.assertEqual(path.read_bytes(), original)

    def test_incomplete_aggregate_requires_explicit_progress_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(BenchmarkContractError, "incomplete"):
                aggregate_suite(self.suite, root, allow_incomplete=False)
            progress = aggregate_suite(self.suite, root, allow_incomplete=True)
            self.assertFalse(progress["complete"])
            self.assertEqual(progress["completed_jobs"], 0)
            self.assertEqual(len(progress["missing_jobs"]), 180)

    def test_scenario_completion_is_independent_of_global_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_new_json_atomic(
                root / "run_contract.json", _fake_run_contract(self.suite)
            )
            for job in self.suite.jobs[:30]:
                _write_new_json_atomic(
                    self.suite.output_path(root, job),
                    _fake_shard(self.suite, job),
                )
            progress = aggregate_suite(self.suite, root, allow_incomplete=True)

        self.assertFalse(progress["complete"])
        self.assertTrue(progress["scenario_results"][0]["complete"])
        self.assertEqual(
            progress["scenario_results"][0]["missing_environment_seeds"], []
        )
        self.assertFalse(progress["scenario_results"][1]["complete"])
        self.assertEqual(
            progress["scenario_results"][1]["missing_environment_seeds"],
            list(self.suite.benchmark_seeds),
        )

    def test_invalid_first_run_contract_is_not_published(self):
        invalid = _fake_run_contract(self.suite)
        invalid["runtime_metadata"]["rvo2_module"] = {
            "path": "/test/wrong-rvo2.so",
            "sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "cfm_mppi.evaluation.eval_socnavgym_suite._current_run_contract",
                return_value=invalid,
            ):
                with self.assertRaisesRegex(BenchmarkContractError, "RVO2"):
                    _ensure_run_contract(root, self.suite, Path("unused"), {})
            self.assertFalse((root / "run_contract.json").exists())

    def test_complete_aggregate_has_scenario_role_and_overall_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_new_json_atomic(
                root / "run_contract.json", _fake_run_contract(self.suite)
            )
            for job in self.suite.jobs:
                path = self.suite.output_path(root, job)
                _write_new_json_atomic(path, _fake_shard(self.suite, job))

            aggregate = aggregate_suite(
                self.suite, root, allow_incomplete=False
            )

        self.assertTrue(aggregate["complete"])
        self.assertEqual(aggregate["completed_jobs"], 180)
        self.assertEqual(aggregate["completed_scenario_seed_pairs"], 180)
        self.assertEqual(len(aggregate["scenario_results"]), 6)
        self.assertEqual(
            aggregate["scenario_results"][0]["planners"]["cfm-mppi-cv"]["episodes"],
            30,
        )
        self.assertEqual(aggregate["overall"]["cfm-mppi-cv"]["episodes"], 180)
        self.assertEqual(aggregate["overall"]["vrc-mppi"]["episodes"], 180)
        self.assertEqual(aggregate["overall"]["cfm-mppi-cv"]["success_rate"], 1.0)
        self.assertEqual(aggregate["overall"]["vrc-mppi"]["collision_rate"], 1.0)
        self.assertEqual(
            aggregate["role_results"]["primary"]["cfm-mppi-cv"]["episodes"],
            90,
        )

    def test_aggregate_rejects_shard_runtime_drift_from_root_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_new_json_atomic(
                root / "run_contract.json", _fake_run_contract(self.suite)
            )
            job = self.suite.job(0)
            shard = _fake_shard(self.suite, job)
            shard["metadata"]["torch"] = "different"
            _write_new_json_atomic(self.suite.output_path(root, job), shard)
            with self.assertRaisesRegex(BenchmarkContractError, "runtime changed"):
                aggregate_suite(self.suite, root, allow_incomplete=True)

    def test_shard_validation_does_not_mutate_document(self):
        job = self.suite.job(1)
        document = _fake_shard(self.suite, job)
        snapshot = deepcopy(document)
        validate_shard_document(document, self.suite, job)
        self.assertEqual(document, snapshot)


if __name__ == "__main__":
    unittest.main()
