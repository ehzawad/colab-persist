"""CLI routing for the v0.4 commands and hardware flags."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from colab_persist import backend, client


class CLITests(unittest.TestCase):
    def run_cli(self, *argv, invoke=None):
        calls = []

        def fake_invoke(name, **arguments):
            calls.append((name, arguments))
            return (invoke or {}).get(name, {})
        out, err = io.StringIO(), io.StringIO()
        with patch.object(client.sys, "argv", ["colab-persist", *argv]), \
             patch.object(client, "invoke", side_effect=fake_invoke), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                client.main()
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return calls, out.getvalue(), err.getvalue(), code

    def test_setup_and_specs(self):
        with patch.object(backend, "setup", return_value={"project_saved": True}) as setup:
            _, out, _, code = self.run_cli("setup", "--project", "demo-project")
        setup.assert_called_once_with("demo-project")
        self.assertEqual((json.loads(out), code), ({"project_saved": True}, 0))
        calls, _, _, _ = self.run_cli("specs", invoke={"runtime_specs": {"eligible": ["L4"]}})
        self.assertEqual(calls, [("runtime_specs", {})])

    def test_start_with_hardware_flags_and_workspace_preparation(self):
        start = {"runtime": "cp-cuda-abc123", "gpu": "V5E1 high-mem", "drive_connected": True}
        calls, _, err, _ = self.run_cli("start", "--tpu", "V5E1", "--high-mem", "--project", "demo",
                                        invoke={"start_runtime": start})
        self.assertEqual(calls, [("start_runtime", {"gpu": None, "tpu": "V5E1", "cpu": False, "high_mem": True}),
                                 ("prepare_workspace", {"project": "demo"})])
        self.assertIn("Runtime cp-cuda-abc123: V5E1 high-mem; Drive connected", err)
        calls, _, _, _ = self.run_cli("start", "--cpu", "--no-mount", invoke={"start_runtime": start})
        self.assertEqual(calls, [("start_runtime", {"gpu": None, "tpu": None, "cpu": True, "high_mem": False})])

    def test_run_detach_and_exit_codes(self):
        with patch.object(backend, "validate_run_inputs"):
            calls, _, _, code = self.run_cli(
                "run", "train.py", "--detach", "--keep-runtime", "--cpu",
                invoke={"start_runtime": {"runtime": "r", "gpu": "CPU"}, "run_script": {"detached": True}})
            self.assertEqual(code, 0)
            self.assertTrue(calls[1][1]["detach"])
            self.assertFalse(calls[1][1]["stop_after"])
            _, _, _, code = self.run_cli("run", "train.py", invoke={
                "start_runtime": {"runtime": "r", "gpu": "L4"}, "run_script": {"exit_code": 3}})
        self.assertEqual(code, 3)

    def test_attach_and_job(self):
        calls, _, _, _ = self.run_cli("attach", "--project", "demo", "--keep-runtime")
        self.assertEqual(calls, [("attach_job", {"project": "demo", "stop_after": False})])
        calls, _, _, _ = self.run_cli("attach")
        self.assertEqual(calls, [("attach_job", {"project": "default", "stop_after": None})])
        calls, out, err, _ = self.run_cli("job", "--project", "demo",
                                          invoke={"job_status": {"state": "running", "log": "epoch 3\n"}})
        self.assertEqual(calls, [("job_status", {"project": "demo"})])
        self.assertEqual(json.loads(out), {"state": "running"})
        self.assertIn("epoch 3", err)

    def test_import_legacy_and_mount_use_the_backend_directly(self):
        with patch.object(backend, "import_legacy", return_value={"imported": []}) as legacy:
            self.run_cli("import-legacy", "--project", "a", "--project", "b", "--all-snapshots",
                         "--legacy-folder", "Old-Folder")
        legacy.assert_called_once_with(["a", "b"], True, "Old-Folder")
        with patch.object(backend, "import_legacy", return_value={}) as legacy:
            self.run_cli("import-legacy")
        legacy.assert_called_once_with(None, False, None)
        with patch.object(backend, "mount_drive", return_value={"drive_connected": True}) as mount:
            calls, _, _, _ = self.run_cli("mount")
        mount.assert_called_once_with()
        self.assertEqual(calls, [])

    def test_errors_exit_nonzero_with_a_message(self):
        with patch.object(backend, "setup", side_effect=RuntimeError("Could not enable APIs")):
            _, _, err, code = self.run_cli("setup", "--project", "demo-project")
        self.assertEqual(code, 1)
        self.assertIn("colab-persist: Could not enable APIs", err)


if __name__ == "__main__":
    unittest.main()
