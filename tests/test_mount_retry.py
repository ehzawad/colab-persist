"""Drive connects through the API with no browser; the one-time legacy mount is retried."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import accounts, backend

from support import EMAIL, BackendFixture


class MountDriveTests(BackendFixture, unittest.TestCase):
    """`colab-persist mount` checks Drive from this computer and pushes a token to the VM."""

    def setUp(self):
        self.use_backend()
        self.run = self.start_patch(patch.object(backend.subprocess, "run"))

    def test_without_a_runtime_no_browser_or_vm_is_needed(self):
        with patch.object(backend, "remote_call") as remote_call:
            result = backend.mount_drive()
        self.assertEqual(result, {"drive_connected": True, "drive_mode": "api", "drive_mounted": False,
                                  "drive_folder": "My Drive/Colab-Persist", "browser_consent_needed": False,
                                  "drive_email": EMAIL})
        remote_call.assert_not_called()
        self.run.assert_not_called()

    def test_with_a_runtime_the_vm_receives_a_token_and_is_checked(self):
        runtime = self.managed()
        with patch.object(backend, "remote_call", return_value={"drive_connected": True,
                                                                "drive_email": EMAIL.upper()}) as remote_call:
            result = backend.mount_drive()
        remote_call.assert_called_once_with("connect", item=runtime)
        self.assertTrue(result["vm_connected"])
        self.run.assert_not_called()

    def test_vm_reaching_another_account_is_refused(self):
        self.managed()
        with patch.object(backend, "remote_call", return_value={"drive_email": "someone-else@example.com"}):
            with self.assertRaisesRegex(RuntimeError, "different Drive account"):
                backend.mount_drive()

    def test_local_drive_account_mismatch_is_refused_before_the_vm(self):
        self.managed()
        self.drive.email = "someone-else@example.com"
        with patch.object(backend, "remote_call") as remote_call:
            with self.assertRaisesRegex(RuntimeError, "Drive account mismatch: expected first@example.com"):
                backend.mount_drive()
        remote_call.assert_not_called()


class LegacyMountTests(BackendFixture, unittest.TestCase):
    """import-legacy mounts Drive the v0.3 way once, with google-colab-cli in a temporary env."""

    def setUp(self):
        self.use_backend()
        self.runtime = self.managed()
        self.cfg = backend.config()
        self.start_patch(patch.object(backend.shutil, "which", return_value="/usr/local/bin/uv"))
        self.commands = []

    def run_mount(self, mounted, returncodes):
        def run(command, env=None, **_kwargs):
            arguments = Path(command[-1])
            self.commands.append({"command": command, "env": env, "arguments": json.loads(arguments.read_text()),
                                  "mode": stat.S_IMODE(arguments.stat().st_mode), "path": arguments})
            return SimpleNamespace(returncode=returncodes.pop(0))
        err = io.StringIO()
        with patch.object(backend.subprocess, "run", side_effect=run), \
             patch.object(backend, "remote_call", side_effect=[{"drive_mounted": m} for m in mounted]), \
             patch.object(backend.time, "sleep") as sleep, contextlib.redirect_stderr(err):
            try:
                result = backend.legacy_mount(self.cfg, self.runtime)
            except RuntimeError as error:
                result = error
        return result, sleep, err.getvalue()

    def test_failed_first_mount_after_consent_is_explained_and_retried_after_a_pause(self):
        result, sleep, err = self.run_mount([False, True], [1, 0])
        self.assertIs(result, True)
        sleep.assert_called_once_with(5)
        self.assertIn("racing the new credentials", err)
        self.assertEqual(len(self.commands), 2)

    def test_third_attempt_can_still_mount(self):
        result, sleep, _ = self.run_mount([False, False, True], [1, 1, 0])
        self.assertIs(result, True)
        self.assertEqual(sleep.call_count, 2)

    def test_gives_up_after_the_last_attempt_without_a_trailing_pause(self):
        result, sleep, _ = self.run_mount([False, False, False], [1, 1, 1])
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Run `colab-persist import-legacy` again", str(result))
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(self.commands), 3)

    def test_cli_success_without_a_real_mount_is_retried(self):
        result, sleep, _ = self.run_mount([False, True], [0, 0])
        self.assertIs(result, True)
        sleep.assert_called_once()

    def test_mount_helper_runs_in_a_temporary_environment_with_private_arguments(self):
        self.run_mount([True], [0])
        call = self.commands[0]
        command = call["command"]
        self.assertEqual(command[:7], ["/usr/local/bin/uv", "run", "--no-project", "--quiet", "--with",
                                       backend.LEGACY_CLI, "python"])
        self.assertTrue(command[7].endswith("legacy_mount.py"))
        self.assertEqual(call["env"]["GOOGLE_APPLICATION_CREDENTIALS"], str(accounts.credentials_path(self.cfg)))
        self.assertEqual(call["mode"], 0o600)
        info = self.runtime["connectionInfo"]
        self.assertEqual(call["arguments"]["url"], info["url"])
        self.assertEqual(call["arguments"]["token"], info["token"])
        self.assertEqual(call["arguments"]["endpoint"], "m-s-cp-cuda-aaa111")
        self.assertEqual(call["arguments"]["name"], "cuda")
        self.assertEqual(call["path"].parent.parent, accounts.account_dir(self.config_dir, EMAIL))
        self.assertFalse(call["path"].exists())  # Temporary directory removed afterwards.
        self.assertNotIn(info["token"], json.dumps(dict(os.environ)))

    def test_requires_uv_and_the_runtime(self):
        with patch.object(backend.shutil, "which", return_value=None), \
             patch.object(backend.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "needs `uv`"):
                backend.legacy_mount(self.cfg, self.runtime)
            run.assert_not_called()
        self.api.runtimes = []
        with patch.object(backend.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "disappeared"):
                backend.legacy_mount(self.cfg, self.runtime)
            run.assert_not_called()


class ImportLegacyTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend(legacy_drive_folder="Training")
        self.runtime = self.managed()

    def test_already_mounted_drive_is_not_mounted_again(self):
        calls = []

        def remote_call(operation, **kwargs):
            calls.append((operation, kwargs))
            return {"drive_mounted": True} if operation == "status" else {"imported": []}
        with patch.object(backend, "remote_call", side_effect=remote_call), \
             patch.object(backend, "legacy_mount") as mount:
            backend.import_legacy(projects=("a", "b"), all_snapshots=1)
        mount.assert_not_called()
        operation, kwargs = calls[-1]
        self.assertEqual(operation, "import_legacy")
        self.assertEqual(kwargs["legacy_folder"], "Training")
        self.assertEqual(kwargs["projects"], ["a", "b"])
        self.assertIs(kwargs["all_snapshots"], True)
        self.assertEqual(kwargs["timeout"], 6 * 3600)

    def test_unmounted_drive_is_mounted_first(self):
        with patch.object(backend, "remote_call", side_effect=[{"drive_mounted": False}, {"imported": []}]) as call, \
             patch.object(backend, "legacy_mount") as mount:
            backend.import_legacy(legacy_folder="Override")
        mount.assert_called_once()
        self.assertEqual(call.call_args.kwargs["legacy_folder"], "Override")
        self.assertIsNone(call.call_args.kwargs["projects"])


if __name__ == "__main__":
    unittest.main()
