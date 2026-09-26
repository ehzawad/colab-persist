import json
from pathlib import Path
import tempfile
import unittest

from colab_persist import remote
from colab_persist.checkpoints import METADATA_FILE, atomic_checkpoint


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "workspace" / "checkpoints"

    def test_published_generation_becomes_visible_as_one_directory(self):
        metadata = {"step": 20, "dataset_manifest_sha256": "a" * 64}
        with atomic_checkpoint(self.root, "step-20", ["adapter.bin", "state/optimizer.bin"], metadata) as staging:
            (staging / "adapter.bin").write_bytes(b"adapter")
            (staging / "state").mkdir()
            (staging / "state" / "optimizer.bin").write_bytes(b"optimizer")
            self.assertEqual(remote.inventory(self.root), {})
            self.assertFalse((self.root / "step-20").exists())
        files = remote.inventory(self.root)
        self.assertEqual(set(files), {"step-20/adapter.bin", "step-20/state/optimizer.bin",
                                     f"step-20/{METADATA_FILE}"})
        document = json.loads((self.root / "step-20" / METADATA_FILE).read_text())
        self.assertEqual(document["schema"], 1)
        self.assertEqual(document["dataset_manifest_sha256"], metadata["dataset_manifest_sha256"])
        self.assertFalse(staging.exists())

    def test_missing_required_file_leaves_only_excluded_partial(self):
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            with atomic_checkpoint(self.root, "step-1", ["adapter", "optimizer"], {}) as staging:
                (staging / "adapter").write_bytes(b"adapter")
        self.assertTrue(staging.is_dir())
        self.assertFalse((self.root / "step-1").exists())
        self.assertEqual(remote.inventory(self.root), {})

    def test_empty_required_file_cannot_be_published(self):
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            with atomic_checkpoint(self.root, "step-1", ["adapter"], {}) as staging:
                (staging / "adapter").touch()
        self.assertEqual(remote.inventory(self.root), {})

    def test_caller_failure_leaves_diagnostic_partial(self):
        with self.assertRaisesRegex(RuntimeError, "training save failed"):
            with atomic_checkpoint(self.root, "step-1", ["adapter"], {}) as staging:
                (staging / "adapter").write_bytes(b"incomplete")
                raise RuntimeError("training save failed")
        self.assertEqual((staging / "adapter").read_bytes(), b"incomplete")
        self.assertEqual(remote.inventory(self.root), {})

    def test_existing_generation_is_never_overwritten(self):
        destination = self.root / "step-1"
        destination.mkdir(parents=True)
        (destination / "precious").write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            with atomic_checkpoint(self.root, "step-1", ["adapter"], {}):
                self.fail("Existing generation should fail before yielding")
        self.assertEqual((destination / "precious").read_bytes(), b"original")

    def test_generation_created_during_context_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            with atomic_checkpoint(self.root, "step-1", ["adapter"], {}) as staging:
                (staging / "adapter").write_bytes(b"new")
                (self.root / "step-1").mkdir()
        self.assertTrue(staging.is_dir())
        self.assertEqual(list((self.root / "step-1").iterdir()), [])

    def test_symlink_root_is_rejected(self):
        target = self.base / "target"
        target.mkdir()
        link = self.base / "linked-root"
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "root cannot be a symbolic link"):
            with atomic_checkpoint(link, "step-1", ["adapter"], {}):
                self.fail("Symlink root should fail before yielding")
        self.assertEqual(list(target.iterdir()), [])

    def test_symlink_anywhere_in_generation_is_rejected(self):
        outside = self.base / "outside"
        outside.write_bytes(b"outside")
        for link_name in ["adapter", "extra"]:
            with self.subTest(link_name=link_name):
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    with atomic_checkpoint(self.root, "step-1", ["adapter"], {}) as staging:
                        if link_name != "adapter":
                            (staging / "adapter").write_bytes(b"adapter")
                        (staging / link_name).symlink_to(outside)
        self.assertEqual(remote.inventory(self.root), {})

    def test_symlink_parent_inside_generation_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "optimizer").write_bytes(b"optimizer")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            with atomic_checkpoint(self.root, "step-1", ["state/optimizer"], {}) as staging:
                (staging / "state").symlink_to(outside, target_is_directory=True)
        self.assertEqual(remote.inventory(self.root), {})

    def test_paths_and_required_files_are_validated_before_writing(self):
        for generation, required in [("../escape", ["adapter"]), ("/absolute", ["adapter"]),
                                     ("step-1", []), ("step-1", ["../escape"]),
                                     ("step-1", ["/absolute"]), ("step-1", ["state/../adapter"]),
                                     ("step-1", [METADATA_FILE]), ("step-1", ["optimizer.tmp"]),
                                     ("step-1", [".config/state.json"])]:
            with self.subTest(generation=generation, required=required):
                with self.assertRaises(ValueError):
                    with atomic_checkpoint(self.root, generation, required, {}):
                        self.fail("Invalid paths should fail before yielding")
        self.assertFalse(self.root.exists())

    def test_metadata_cannot_replace_publication_fields(self):
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            with atomic_checkpoint(self.root, "step-1", ["adapter"], {"schema": 99}):
                self.fail("Reserved metadata should fail before yielding")


if __name__ == "__main__":
    unittest.main()
