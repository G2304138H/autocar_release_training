"""Dependency-light checks for the one/two/four-view evaluation launcher."""

import json
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

from scripts.evaluate_variable_views_npz import build_parser, evaluation_configs, main


class VariableViewEvaluationLauncherTests(unittest.TestCase):
    def _arguments(self, root: Path) -> list[str]:
        checkpoint = root / "best.ckpt"
        checkpoint.touch()
        (root / "projections").mkdir()
        (root / "voxels").mkdir()
        (root / "splits.json").write_text(
            json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
            encoding="utf-8",
        )
        return [
            "--checkpoint", str(checkpoint),
            "--projection-source", str(root / "projections"),
            "--voxel-source", str(root / "voxels"),
            "--split-json", str(root / "splits.json"),
            "--output-root", str(root / "results"),
        ]

    def test_configs_use_first_views_and_identical_case_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            arguments = self._arguments(Path(temporary_directory))
            plans = evaluation_configs(build_parser().parse_args(arguments))
            self.assertEqual(
                [plan[1]["evaluation_view_indices"] for plan in plans],
                [[0], [0, 1], [0, 1, 2, 3]],
            )
            self.assertEqual(
                [plan[1]["eval_num_views"] for plan in plans], [1, 2, 4]
            )
            self.assertEqual(
                {plan[1]["split_json_path"] for plan in plans},
                {str((Path(temporary_directory) / "splits.json").resolve())},
            )
            self.assertEqual(
                {plan[1]["eval_split"] for plan in plans}, {"val_test"}
            )
            self.assertEqual(
                len({plan[1]["eval_output_dir"] for plan in plans}), 3
            )

    def test_dry_run_writes_three_evaluator_configs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(main(self._arguments(root) + ["--dry-run"]), 0)
            for count in (1, 2, 4):
                path = root / "results" / "configs" / f"views_{count}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["evaluation_view_indices"], list(range(count)))
                self.assertEqual(config["evaluation_mode"], "paper_metric")


    def test_child_output_is_logged_and_failure_propagates(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arguments = self._arguments(root) + ["--view-counts", "1"]

            def child(command, **kwargs):
                self.assertEqual(command[1:4], ["-u", "-m", "src.eval_npz"])
                return real_popen(
                    [sys.executable, "-u", "-c",
                     "import sys; print('stage started'); print('failure detail', file=sys.stderr); sys.exit(3)"],
                    **kwargs,
                )

            with patch("scripts.evaluate_variable_views_npz.subprocess.Popen", side_effect=child):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    main(arguments)
            self.assertEqual(error.exception.returncode, 3)
            log = (root / "results" / "logs" / "views_1.log").read_text()
            self.assertIn("stage started", log)
            self.assertIn("failure detail", log)

    def test_prediction_only_skips_metric_mode_for_every_view_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            arguments = self._arguments(Path(temporary_directory))
            args = build_parser().parse_args(arguments + ["--prediction-only"])
            plans = evaluation_configs(args)
            self.assertEqual(
                [config["evaluation_mode"] for _, config in plans],
                ["prediction"] * 3,
            )


if __name__ == "__main__":
    unittest.main()
