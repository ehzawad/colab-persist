import io
import json
from pathlib import Path
import tarfile
import tempfile
import contextlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import remote, backend, client


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.mount = self.base / "drive"
        (self.mount / "MyDrive").mkdir(parents=True)
        self.store = remote.Store(self.base / "vm1", self.mount)
        self.mount_patch = patch("colab_persist.remote.os.path.ismount", return_value=True)
        self.mount_patch.start()
        self.addCleanup(self.mount_patch.stop)
        self.store.prepare("demo")
        self.workspace = self.store.workspace("demo")

    def test_snapshot_survives_replacement_workspace(self):
        (self.workspace / "source.py").write_text("print('hello GPU')\n")
        (self.workspace / "outputs" / "weights.bin").write_bytes(bytes(range(256)) * 40)
        manifest = self.store.snapshot("demo")
        replacement = remote.Store(self.base / "vm2", self.mount)
        result = replacement.prepare("demo")
        self.assertEqual(result["restored"]["snapshot"], manifest["snapshot"])
        self.assertEqual(remote.inventory(replacement.workspace("demo")), remote.inventory(self.workspace))

    def test_unchanged_snapshot_is_reused(self):
        (self.workspace / "data").write_text("stable")
        first = self.store.snapshot("demo")
        second = self.store.snapshot("demo")
        self.assertEqual(first["snapshot"], second["snapshot"])
        self.assertTrue(second["reused"])

    def test_large_workspace_rejected_before_any_content_is_read(self):
        (self.workspace / "accidental-dataset.bin").write_bytes(b"123456")
        self.store.snapshot_limit = 5
        with patch.object(remote, "digest") as hashing:
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                self.store.snapshot("demo")
            hashing.assert_not_called()

    def test_snapshot_preserves_previous_generation_when_disk_is_full(self):
        first = self.store.snapshot("demo")
        (self.workspace / "changed").write_text("new")
        with patch.object(remote.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient scratch"):
                self.store.snapshot("demo")
        self.assertEqual(self.store.snapshots("demo")[0]["snapshot"], first["snapshot"])

    def test_restore_checks_space_before_reading_archive(self):
        self.store.snapshot("demo")
        replacement = remote.Store(self.base / "vm2", self.mount)
        with patch.object(remote.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
             patch.object(remote, "digest") as hashing:
            with self.assertRaisesRegex(RuntimeError, "Insufficient scratch"):
                replacement.restore("demo")
            hashing.assert_not_called()

    def test_failed_source_size_check_never_uploads(self):
        source = self.base / "local-source"
        source.mkdir()
        (source / "train.py").write_text("pass")
        (source / "large.data").write_bytes(b"123456")
        with patch.object(backend, "config", return_value={"snapshot_limit_bytes": 5}), \
             patch.object(backend.subprocess, "run") as command:
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                backend.upload_source(object(), "demo", source / "train.py", source)
            command.assert_not_called()

    def test_data_and_model_caches_are_outside_snapshot(self):
        script = self.workspace / "paths.py"
        script.write_text("import os, pathlib\n"
                          "for key in ('COLAB_DATASET_CACHE', 'HF_HOME'):\n"
                          " p = pathlib.Path(os.environ[key]); p.mkdir(parents=True, exist_ok=True)\n"
                          " (p / 'cached-weights.bin').write_text('disposable')\n"
                          " print(str(p))\n")
        with patch.dict(remote.os.environ, {}, clear=True):
            result = self.store.run("demo", "paths.py", [], interval=10)
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(any("cached-weights" in name for name in result["checkpoint"]["files"]))
        self.assertTrue((self.base / "colab-persist-cache" / "huggingface" / "cached-weights.bin").exists())

    def test_corruption_cannot_be_restored(self):
        (self.workspace / "data").write_text("important")
        manifest = self.store.snapshot("demo")
        (self.store.root / "demo" / (manifest["snapshot"] + ".tar.gz")).write_bytes(b"corrupt")
        replacement = remote.Store(self.base / "vm2", self.mount)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            replacement.restore("demo")
        self.assertFalse(replacement.workspace("demo").exists())

    def test_restore_never_overwrites_existing_work(self):
        self.store.snapshot("demo")
        (self.workspace / "precious").write_text("uncommitted")
        with self.assertRaisesRegex(RuntimeError, "never overwritten"):
            self.store.restore("demo")
        self.assertEqual((self.workspace / "precious").read_text(), "uncommitted")

    def test_credentials_and_links_are_excluded(self):
        (self.workspace / ".env").write_text("SECRET=example")
        (self.workspace / ".ssh").mkdir()
        (self.workspace / ".ssh" / "private").write_text("sensitive")
        outside = self.base / "outside"
        outside.write_text("not part of project")
        (self.workspace / "linked").symlink_to(outside)
        (self.workspace / "valid.py").write_text("print(1)")
        self.assertEqual(set(self.store.snapshot("demo")["files"]), {"valid.py"})

    def test_incomplete_archive_does_not_replace_good_snapshot(self):
        first = self.store.snapshot("demo")
        (self.store.root / "demo" / "999999.tar.gz.partial").write_bytes(b"interrupted")
        self.assertEqual(self.store.snapshots("demo")[0]["snapshot"], first["snapshot"])

    def test_missing_mount_fails_before_creating_cloud_lookalike(self):
        with patch("colab_persist.remote.os.path.ismount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                self.store.snapshot("demo")
        self.assertFalse(self.store.root.exists())

    def test_malicious_archive_cannot_escape_destination(self):
        archive = self.base / "malicious.tar.gz"
        with tarfile.open(archive, "w:gz") as target:
            entry = tarfile.TarInfo("../escaped")
            entry.size = 4
            target.addfile(entry, io.BytesIO(b"evil"))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            remote.extract_archive(archive, self.workspace)
        self.assertFalse((self.workspace.parent / "escaped").exists())

    def test_failed_script_still_checkpoints_error_outputs(self):
        (self.workspace / "fail.py").write_text("print('useful error log'); raise SystemExit(7)\n")
        result = self.store.run("demo", "fail.py", [], interval=10)
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("useful error log", result["log_tail"])
        self.assertIn("outputs/last-run.json", result["checkpoint"]["files"])

    def test_active_job_blocks_stop(self):
        with self.store.project_lock("demo"):
            with self.assertRaisesRegex(RuntimeError, "active managed job"):
                self.store.seal()

    def test_changed_files_after_flush_block_shutdown(self):
        saved = self.store.snapshot("demo")
        remote.write_json(self.store.scratch / "flush-receipt.json",
                          {"drive_flushed": True, "checkpoints": [saved]})
        (self.workspace / "new-output").write_text("not saved")
        with patch("colab_persist.remote.os.path.ismount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                self.store.seal()

    def test_unmanaged_write_during_flush_prevents_success_receipt(self):
        mounted = [True]
        def flush(**kwargs):
            (self.workspace / "late-write").write_text("unsaved")
            mounted[0] = False
        colab = SimpleNamespace(drive=SimpleNamespace(flush_and_unmount=flush))
        with patch.dict(sys.modules, {"google.colab": colab}), \
             patch("colab_persist.remote.os.path.ismount", side_effect=lambda _: mounted[0]):
            with self.assertRaisesRegex(RuntimeError, "changed while"):
                self.store.seal()
        self.assertFalse((self.store.scratch / "flush-receipt.json").exists())


class ShutdownTests(unittest.TestCase):
    def test_changing_snapshot_cap_preserves_existing_runtime_and_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "custom-key"
            key.write_text("test key fixture")
            original = {"expected_email": "test@example.com", "gpu": "A100", "session": "training",
                        "ssh_identity": str(key), "drive_folder": "Training", "ssh_alias_config": "custom.conf"}
            remote.write_json(root / "config.json", original)
            args = SimpleNamespace(email="test@example.com", gpu=None, session=None, key=None,
                                   ssh_config=None, snapshot_limit_gib=8)
            with patch.object(backend, "CONFIG_DIR", root), patch.object(client.subprocess, "run") as command:
                client.configure(args)
                command.assert_not_called()
            self.assertEqual(json.loads((root / "config.json").read_text()),
                             {**original, "snapshot_limit_bytes": 8 * remote.GIB})

    def test_deployed_engine_can_import_dataset_and_checkpoint_helpers(self):
        calls = []
        with patch.object(backend, "ssh_command", return_value=["unused"]), \
             patch.object(backend, "checked", side_effect=lambda args, **kw: calls.append(kw.get("data"))):
            backend.install_engine(object())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()
            with tarfile.open(fileobj=io.BytesIO(calls[0])) as archive:
                archive.extractall(target, filter="data")
            program = (f"import sys; sys.path.insert(0, {str(target)!r}); "
                       "from colab_persist.datasets import ShardCache; "
                       "from colab_persist.checkpoints import atomic_checkpoint; "
                       "from colab_persist import remote; print(remote.__file__)")
            result = backend.subprocess.run([sys.executable, "-I", "-c", program],
                                            capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(target / "colab_persist" / "remote.py"))

    def test_mount_checks_filesystem_even_when_cli_returns_zero(self):
        with patch.object(backend, "operation_lock", side_effect=contextlib.nullcontext), \
             patch.object(backend, "config", return_value={}), \
             patch.object(backend, "require_session", return_value=SimpleNamespace(name="cuda")), \
             patch.object(backend, "colab_command", return_value=["unused"]), \
             patch.object(backend, "remote_call", side_effect=[
                 {"drive_mounted": False}, {"drive_mounted": False}, {"drive_mounted": True}]), \
             patch.object(backend.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as command:
            self.assertTrue(backend.mount_drive()["drive_mounted"])
            self.assertEqual(command.call_count, 2)

    def test_no_shutdown_after_flush_failure(self):
        with patch.object(backend, "kernel_call", side_effect=RuntimeError("Drive flush failed")):
            with patch.object(backend, "checked") as command:
                with self.assertRaisesRegex(RuntimeError, "flush failed"):
                    backend._save_and_stop(object(), True)
                command.assert_not_called()

    def test_no_shutdown_without_completion_receipt(self):
        with patch.object(backend, "kernel_call", return_value={"drive_flushed": False}):
            with patch.object(backend, "checked") as command:
                with self.assertRaisesRegex(RuntimeError, "Shutdown refused"):
                    backend._save_and_stop(object(), True)
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
