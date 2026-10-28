from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from project_paths import artifact_path, relative_path, require_repo_root


class ProjectPathTests(unittest.TestCase):
    def test_relative_path_accepts_repo_relative_slash_path(self) -> None:
        self.assertEqual(
            relative_path("datasets/pt/ptbxl_testset.pt"),
            Path("datasets/pt/ptbxl_testset.pt"),
        )

    def test_relative_path_rejects_absolute_parent_and_backslash_paths(self) -> None:
        invalid = (
            "/kaggle/input/data.pt",
            "D:/ECG/data.pt",
            "D:ECG/data.pt",
            "D:\\ECG\\data.pt",
            "../input/data.pt",
            "datasets\\pt\\data.pt",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                relative_path(value)

    def test_artifact_path_confines_outputs(self) -> None:
        self.assertEqual(
            artifact_path("artifacts/metrics/E0/results.csv"),
            Path("artifacts/metrics/E0/results.csv"),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            artifact_path("results/E0.csv")

    def test_require_repo_root_uses_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            marker = root / "paper_v2/RUNBOOK_CHECKLIST.csv"
            marker.parent.mkdir(parents=True)
            marker.write_text("marker\n", encoding="utf-8")
            self.assertEqual(require_repo_root(root), root.resolve())

    def test_require_repo_root_rejects_other_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(RuntimeError):
                require_repo_root(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
