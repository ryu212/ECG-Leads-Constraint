from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_kaggle_dataset import (
    MANIFEST_RELATIVE_PATH,
    SOURCE_REQUIRED_FILES,
    PreparationError,
    audit_source,
    collect_plan,
    prepare_snapshot,
    verify_snapshot,
)


class PrepareKaggleDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.source = self.base / "WRFB_MCMA"
        self.source.mkdir()

        for index, relative in enumerate(SOURCE_REQUIRED_FILES, start=1):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"required-{index}".encode("utf-8"))

        (self.source / "v13").mkdir(exist_ok=True)
        (self.source / "v13/model.py").write_text("MODEL = True\n", encoding="utf-8")
        (self.source / "v13/results").mkdir(parents=True, exist_ok=True)
        (self.source / "v13/results/result.xlsx").write_bytes(b"valid-result")
        (self.source / "v13/results/~$locked.xlsx").write_bytes(b"lock")
        (self.source / "__pycache__").mkdir()
        (self.source / "__pycache__/model.pyc").write_bytes(b"cache")
        (self.source / "artifacts/smoke").mkdir(parents=True)
        (self.source / "artifacts/smoke/trial.pt").write_bytes(b"trial")
        (self.source / "artifacts/manifests").mkdir(parents=True)
        (self.source / "artifacts/manifests/seed_training_plan.csv").write_text(
            "version,seeds\nv13,42\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_audit_keeps_payload_and_excludes_unambiguous_junk(self) -> None:
        result = audit_source(self.source)
        self.assertEqual(result["status"], "PASS")
        plan = {record.path: record for record in collect_plan(self.source)}
        self.assertEqual(plan["v13/model.py"].action, "include")
        self.assertEqual(plan["v13/results/result.xlsx"].action, "include")
        self.assertEqual(plan["v13/results/~$locked.xlsx"].action, "exclude")
        self.assertEqual(plan["__pycache__/"].action, "exclude")
        self.assertEqual(plan["artifacts/smoke/"].action, "exclude")
        self.assertEqual(
            plan["artifacts/manifests/seed_training_plan.csv"].action, "include"
        )

    def test_missing_required_asset_fails(self) -> None:
        (self.source / SOURCE_REQUIRED_FILES[0]).unlink()
        with self.assertRaises(PreparationError):
            audit_source(self.source)

    def test_prepare_and_verify_snapshot(self) -> None:
        package = self.base / "release"
        summary = prepare_snapshot(self.source, package)
        snapshot = package / "WRFB_MCMA"
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue((snapshot / "v13/model.py").is_file())
        self.assertTrue((snapshot / "v13/results/result.xlsx").is_file())
        self.assertTrue((snapshot / "external/code_test/gold_standard.csv").is_file())
        self.assertFalse((snapshot / "v13/results/~$locked.xlsx").exists())
        self.assertFalse((snapshot / "__pycache__").exists())
        self.assertTrue(
            (snapshot / "artifacts/manifests/seed_training_plan.csv").is_file()
        )
        self.assertTrue((snapshot / MANIFEST_RELATIVE_PATH).is_file())
        self.assertEqual(verify_snapshot(snapshot)["status"], "PASS")

        with (snapshot / MANIFEST_RELATIVE_PATH).open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            paths = [row["path"] for row in csv.DictReader(handle)]
        self.assertNotIn(MANIFEST_RELATIVE_PATH.as_posix(), paths)
        self.assertEqual(paths, sorted(paths))

    def test_existing_snapshot_is_never_overwritten(self) -> None:
        package = self.base / "release"
        prepare_snapshot(self.source, package)
        with self.assertRaises(PreparationError):
            prepare_snapshot(self.source, package)

    def test_verify_detects_tampering(self) -> None:
        package = self.base / "release"
        prepare_snapshot(self.source, package)
        snapshot = package / "WRFB_MCMA"
        (snapshot / "v13/model.py").write_text("TAMPERED = True\n", encoding="utf-8")
        with self.assertRaises(PreparationError):
            verify_snapshot(snapshot)

    def test_verify_detects_unexpected_file(self) -> None:
        package = self.base / "release"
        prepare_snapshot(self.source, package)
        snapshot = package / "WRFB_MCMA"
        (snapshot / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaises(PreparationError):
            verify_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
