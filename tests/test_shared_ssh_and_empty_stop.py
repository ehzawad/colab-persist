import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import backend, remote


class EmptyRuntimeStopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = remote.Store(self.base / "vm", self.base / "drive")

    def test_runtime_without_workspaces_or_drive_has_nothing_to_save(self):
        receipt = self.store.seal()
        self.assertTrue(receipt["nothing_to_save"])
        self.assertFalse(receipt["drive_flushed"])
        self.assertEqual(receipt["checkpoints"], [])

    def test_workspace_without_drive_still_refuses(self):
        (self.base / "drive" / "MyDrive").mkdir(parents=True)
        with patch("colab_persist.remote.os.path.ismount", return_value=True):
            self.store.prepare("demo")
        with self.assertRaisesRegex(RuntimeError, "Drive is not mounted"):
            self.store.seal()

    def test_nothing_to_save_stops_the_runtime_without_a_drive_receipt(self):
        item = SimpleNamespace(name="cuda", endpoint="gpu-a")
        receipt = {"drive_flushed": False, "nothing_to_save": True, "checkpoints": []}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(backend, "CONFIG_DIR", Path(directory)), \
             patch.object(backend, "kernel_call", return_value=receipt), \
             patch.object(backend, "require_session", return_value=item), \
             patch.object(backend, "colab_command", return_value=["colab"]), \
             patch.object(backend, "config", return_value={}), \
             patch.object(backend.accounts, "environment", return_value={}), \
             patch.object(backend, "checked") as command:
            result = backend._save_and_stop(item, True)
            self.assertFalse((Path(directory) / "last-save.json").exists())
        command.assert_called_once()
        self.assertEqual(command.call_args.args[0], ["colab", "stop", "-s", "cuda"])
        self.assertTrue(result["stopped"])
        self.assertTrue(result["nothing_to_save"])
        self.assertFalse(result["drive_flushed"])


class RemoteErrorTests(unittest.TestCase):
    def test_engine_reports_why_an_operation_failed(self):
        output = io.StringIO()
        real_store = remote.Store
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output), \
             patch.object(remote, "Store", side_effect=lambda **kw: real_store(
                 scratch=Path(directory) / "vm", mount=Path(directory) / "drive", **kw)):
            with self.assertRaisesRegex(RuntimeError, "Drive is not mounted"):
                remote.dispatch({"operation": "prepare", "project": "demo"})
        line = output.getvalue().strip().splitlines()[-1]
        self.assertTrue(line.startswith(remote.ERROR_PREFIX))
        detail = json.loads(line[len(remote.ERROR_PREFIX):])
        self.assertEqual(detail["operation"], "prepare")
        self.assertIn("Drive is not mounted", detail["error"])

    def test_failure_names_the_operation_and_its_reason(self):
        raw = "Traceback...\n" + remote.ERROR_PREFIX + json.dumps({"operation": "status", "error": "disk full"})
        with self.assertRaises(RuntimeError) as status_error:
            backend.parse_result(raw, "status")
        self.assertIn("'status'", str(status_error.exception))
        self.assertIn("disk full", str(status_error.exception))
        self.assertNotIn("NOT stopped", str(status_error.exception))
        with self.assertRaisesRegex(RuntimeError, "Runtime was NOT stopped"):
            backend.parse_result(raw, "seal")

    def test_results_still_parse(self):
        raw = "noise\n" + remote.RESULT_PREFIX + json.dumps({"drive_mounted": True})
        self.assertEqual(backend.parse_result(raw, "status"), {"drive_mounted": True})


class SharedSSHTests(unittest.TestCase):
    def test_commands_share_one_connection_per_runtime(self):
        with patch.object(backend, "config", return_value={}):
            first = backend.ssh_command(SimpleNamespace(endpoint="gpu-a"))
            second = backend.ssh_command(SimpleNamespace(endpoint="gpu-b"))
        self.assertIn("ControlMaster=auto", first)
        self.assertIn("ControlPersist=30", first)
        paths = [arg for arg in first + second if arg.startswith("ControlPath=")]
        self.assertEqual(paths, ["ControlPath=" + str(backend.SSH_CONTROL_DIR / "cm-colab-gpu-a"),
                                 "ControlPath=" + str(backend.SSH_CONTROL_DIR / "cm-colab-gpu-b")])

    def test_alias_shares_the_same_connection(self):
        item = SimpleNamespace(endpoint="gpu-a")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "colab-cuda.conf"
            with patch.object(backend, "config", return_value={"ssh_alias_config": str(target)}):
                backend.write_ssh_alias(item)
            text = target.read_text()
        self.assertIn("ControlMaster auto", text)
        self.assertIn(f'ControlPath "{backend.control_path(item)}"', text)

    def test_busy_ssh_slot_is_explained(self):
        busy = SimpleNamespace(returncode=255, stdout=b"",
                               stderr=b"[colab] Already-active SSH session (HTTP 429): another `colab ssh` is connected")
        with patch.object(backend.subprocess, "run", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "one SSH session per runtime.*ssh -O exit colab-cuda"):
                backend.checked(["ssh", "colab-runtime", "true"])


if __name__ == "__main__":
    unittest.main()
