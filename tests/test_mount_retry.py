import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import backend


class MountRetryTests(unittest.TestCase):
    def patches(self, mounted, returncodes):
        # remote_call("status") runs once up front and again only after a drivemount that exits 0.
        return [patch.object(backend, "operation_lock", side_effect=contextlib.nullcontext),
                patch.object(backend, "config", return_value={}),
                patch.object(backend, "require_session", return_value=SimpleNamespace(name="cuda")),
                patch.object(backend, "colab_command", return_value=["colab"]),
                patch.object(backend.accounts, "environment", return_value={}),
                patch.object(backend, "remote_call", side_effect=[{"drive_mounted": m} for m in mounted]),
                patch.object(backend.subprocess, "run",
                             side_effect=[SimpleNamespace(returncode=r) for r in returncodes])]

    def run_mount(self, mounted, returncodes):
        err = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in self.patches(mounted, returncodes):
                stack.enter_context(p)
            sleep = stack.enter_context(patch.object(backend.time, "sleep"))
            stack.enter_context(contextlib.redirect_stderr(err))
            try:
                return backend.mount_drive(), sleep, err.getvalue()
            except RuntimeError as error:
                return error, sleep, err.getvalue()

    def test_failed_first_mount_after_consent_is_explained_and_retried_after_a_pause(self):
        result, sleep, err = self.run_mount([False, True], [1, 0])
        self.assertEqual(result, {"drive_mounted": True})
        sleep.assert_called_once_with(backend.MOUNT_RETRY_PAUSE_S)
        self.assertIn("racing the new credentials", err)
        self.assertIn("attempt 2 of 3", err)

    def test_third_attempt_can_still_mount(self):
        result, sleep, _ = self.run_mount([False, True], [1, 1, 0])
        self.assertEqual(result, {"drive_mounted": True})
        self.assertEqual(sleep.call_count, 2)

    def test_gives_up_after_the_last_attempt_without_a_trailing_pause(self):
        result, sleep, _ = self.run_mount([False], [1, 1, 1])
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Run `colab-persist mount` again", str(result))
        self.assertEqual(sleep.call_count, 2)

    def test_cli_success_without_a_real_mount_is_retried(self):
        result, sleep, _ = self.run_mount([False, False, True], [0, 0])
        self.assertEqual(result, {"drive_mounted": True})
        sleep.assert_called_once()

    def test_already_mounted_drive_is_left_alone(self):
        result, sleep, _ = self.run_mount([True], [])
        self.assertEqual(result, {"drive_mounted": True})
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
