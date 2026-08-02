import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cfm_mppi.evaluation.eval_socnavgym import (
    _load_model,
    parse_seed_spec,
)


class SocNavGymEvaluationCLITest(unittest.TestCase):
    def test_parses_seed_lists_and_ranges(self):
        self.assertEqual(
            parse_seed_spec("1,3:7,11:6:-2"),
            (1, 3, 4, 5, 6, 11, 9, 7),
        )
        for invalid in ("", "1,,2", "1:3:0", "1,1", "3:3", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_seed_spec(invalid)

    def test_missing_checkpoint_requires_explicit_smoke_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "checkpoint.pth"
            with self.assertRaises(FileNotFoundError):
                _load_model(
                    missing,
                    device=torch.device("cpu"),
                    allow_random_model=False,
                )
            model, loaded = _load_model(
                missing,
                device=torch.device("cpu"),
                allow_random_model=True,
            )

        self.assertFalse(loaded)
        self.assertFalse(model.training)

    def test_loads_legacy_namespace_checkpoint_with_weights_only(self):
        source_model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            torch.save(
                {
                    "model": source_model.state_dict(),
                    "args": argparse.Namespace(output_dir="ignored"),
                },
                checkpoint,
            )
            with mock.patch(
                "cfm_mppi.evaluation.eval_socnavgym.TransformerModel",
                return_value=torch.nn.Linear(1, 1),
            ):
                model, loaded = _load_model(
                    checkpoint,
                    device=torch.device("cpu"),
                    allow_random_model=False,
                )

        self.assertTrue(loaded)
        self.assertFalse(model.training)
        torch.testing.assert_close(model.weight, source_model.weight)
        torch.testing.assert_close(model.bias, source_model.bias)


if __name__ == "__main__":
    unittest.main()
