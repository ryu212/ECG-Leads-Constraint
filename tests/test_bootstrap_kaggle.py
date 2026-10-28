from __future__ import annotations

import csv
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from tools.bootstrap_kaggle import (
    ARTIFACT_RELATIVE_PATHS,
    MANIFEST_RELATIVE_PATH,
    MARKER_RELATIVE_PATH,
    REPORT_RELATIVE_PATH,
    REQUIRED_ASSET_FILES,
    BootstrapError,
    bootstrap,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BootstrapKaggleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.input_base = self.base / "kaggle/input"
        self.input_root = self.input_base / "snapshot/WRFB_MCMA"
        self.working_root = self.base / "kaggle/working/WRFB_MCMA"
        self.input_root.mkdir(parents=True)
        self.original_cwd = Path.cwd()
        self.added_sys_paths: list[str] = []
        self._create_valid_input()

    def tearDown(self) -> None:
        os.chdir(self.original_cwd)
        working_text = str(self.working_root.resolve())
        while working_text in sys.path:
            sys.path.remove(working_text)
        self.temporary_directory.cleanup()

    def _write(self, relative: Path | str, content: bytes) -> Path:
        path = self.input_root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _create_valid_input(self) -> None:
        for index, relative in enumerate(REQUIRED_ASSET_FILES, start=1):
            self._write(relative, f"asset-{index}".encode("utf-8"))
        self._write("requirements.txt", b"torch\n")
        self._write("v13/model.py", b"MODEL = True\n")
        self._write("v13/results/legacy.xlsx", b"legacy-result")
        self._write("tools/bootstrap_kaggle.py", b"# bootstrap source\n")
        self._write(
            "artifacts/manifests/seed_training_plan.csv",
            b"version,seeds\nv13,42\n",
        )
        self._write(
            "external/automatic-ecg-diagnosis/model.py", b"CLASSIFIER = True\n"
        )
        self._write(
            "external/automatic-ecg-diagnosis/outputs/old.pdf", b"generated-output"
        )
        self._write("__pycache__/model.pyc", b"cache")

        payload = sorted(
            path
            for path in self.input_root.rglob("*")
            if path.is_file() and path.relative_to(self.input_root) != MANIFEST_RELATIVE_PATH
        )
        manifest = self.input_root / MANIFEST_RELATIVE_PATH
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
            writer.writeheader()
            for path in payload:
                writer.writerow(
                    {
                        "path": path.relative_to(self.input_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

    def _read_report(self) -> list[dict[str, str]]:
        with (self.working_root / REPORT_RELATIVE_PATH).open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            return list(csv.DictReader(handle))

    def test_bootstrap_copies_source_and_creates_relative_links(self) -> None:
        result = bootstrap(self.input_base, self.working_root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["relative_symlink_count"], 4)
        self.assertEqual(Path.cwd(), self.working_root.resolve())
        self.assertEqual(sys.path.count(str(self.working_root.resolve())), 1)
        self.assertTrue((self.working_root / "v13/model.py").is_file())
        self.assertTrue((self.working_root / "tools/bootstrap_kaggle.py").is_file())
        self.assertFalse((self.working_root / "v13/results").exists())
        self.assertFalse((self.working_root / "__pycache__").exists())
        self.assertFalse(
            (self.working_root / "external/automatic-ecg-diagnosis/outputs").exists()
        )
        self.assertEqual(result["artifact_directory_count"], 9)
        self.assertEqual(result["artifact_directory_created_count"], 8)
        for relative in ARTIFACT_RELATIVE_PATHS:
            directory = self.working_root / relative
            self.assertTrue(directory.is_dir())
            self.assertFalse(directory.is_symlink())

        for relative in ("datasets", "checkpoints", "external/code_test"):
            link = self.working_root / relative
            self.assertTrue(link.is_symlink())
            self.assertFalse(os.path.isabs(os.readlink(link)))
        model_link = self.working_root / "external/model.hdf5"
        self.assertTrue(model_link.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(model_link)))

        statuses = {row["status"] for row in self._read_report()}
        self.assertEqual(statuses, {"PASS"})
        artifact_rows = [
            row for row in self._read_report() if row["record_type"] == "artifact_directory"
        ]
        self.assertEqual(len(artifact_rows), 9)

    def test_no_marker_fails_and_writes_report(self) -> None:
        (self.input_root / MARKER_RELATIVE_PATH).unlink()
        with self.assertRaises(BootstrapError):
            bootstrap(self.input_base, self.working_root)
        rows = self._read_report()
        self.assertEqual(rows[-1]["status"], "FAIL")
        self.assertIn("exactly one", rows[-1]["detail"])

    def test_multiple_markers_fail_and_write_report(self) -> None:
        other = self.input_base / "other/WRFB_MCMA" / MARKER_RELATIVE_PATH
        other.parent.mkdir(parents=True)
        other.write_text("marker\n", encoding="utf-8")
        with self.assertRaises(BootstrapError):
            bootstrap(self.input_base, self.working_root)
        self.assertEqual(self._read_report()[-1]["status"], "FAIL")

    def test_missing_asset_fails_before_source_copy(self) -> None:
        (self.input_root / "datasets/pt/cpsc2018.pt").unlink()
        with self.assertRaises(BootstrapError):
            bootstrap(self.input_base, self.working_root)
        self.assertFalse((self.working_root / "v13/model.py").exists())
        failures = [row for row in self._read_report() if row["status"] == "FAIL"]
        self.assertTrue(any(row["path"] == "datasets/pt/cpsc2018.pt" for row in failures))

    def test_manifest_size_mismatch_fails(self) -> None:
        with (self.input_root / "datasets/pt/cpsc2018.pt").open("ab") as handle:
            handle.write(b"changed-size")
        with self.assertRaises(BootstrapError):
            bootstrap(self.input_base, self.working_root)
        self.assertTrue(
            any(row["detail"] == "size_mismatch" for row in self._read_report())
        )

    def test_optional_sha256_check_detects_same_size_tampering(self) -> None:
        asset = self.input_root / "datasets/pt/cpsc2018.pt"
        original = asset.read_bytes()
        asset.write_bytes(b"X" * len(original))
        with self.assertRaises(BootstrapError):
            bootstrap(self.input_base, self.working_root, verify_sha256=True)
        self.assertTrue(
            any(row["detail"] == "sha256_mismatch" for row in self._read_report())
        )

    def test_rerun_is_idempotent_and_preserves_output(self) -> None:
        bootstrap(self.input_base, self.working_root)
        os.chdir(self.original_cwd)
        output = self.working_root / "artifacts/metrics/user_result.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("result\n", encoding="utf-8")
        second = bootstrap(self.input_base, self.working_root)
        self.assertEqual(second["status"], "PASS")
        self.assertEqual(output.read_text(encoding="utf-8"), "result\n")

    def test_non_symlink_asset_destination_is_never_replaced(self) -> None:
        conflict = self.working_root / "datasets"
        conflict.mkdir(parents=True)
        marker = conflict / "user-file.txt"
        marker.write_text("keep\n", encoding="utf-8")
        with self.assertRaises(BootstrapError):
            bootstrap(self.input_base, self.working_root)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")
        self.assertEqual(self._read_report()[-1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
