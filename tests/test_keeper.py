"""The background helper on this computer that keeps a running job's Drive access fresh."""
import fcntl
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from colab_persist import backend, keeper

from support import BackendFixture

RUNNING = {"workspaces": {"train": {"active": True, "job": "running"}}}
IDLE = {"workspaces": {"train": {"active": False, "job": "finished"}}}


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class KeeperCase(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.runtime = self.managed()
        self.clock = Clock()
        self.logs = []
        self.start_patch(patch.object(keeper, "log", side_effect=self.logs.append))

    def run_keeper(self, sleep=None):
        return keeper.run(clock=self.clock, sleep=sleep or self.clock.sleep)

    def statuses(self, *items):
        return self.start_patch(patch.object(backend, "remote_call", side_effect=items))


class RunTests(KeeperCase):
    def test_renews_every_ten_minutes_while_a_job_runs_then_stops_by_itself(self):
        call = self.statuses(RUNNING, RUNNING, RUNNING, IDLE)
        start = self.clock.now
        self.assertEqual(self.run_keeper(), "idle")
        self.assertEqual([c.args for c in call.call_args_list], [("status",)] * 4)
        self.assertEqual(self.clock.now - start, 4 * keeper.RENEW_SECONDS)
        self.assertIn("No managed job is active; stopping.", self.logs)

    def test_fifteen_hours_of_work_never_leave_the_vm_without_a_valid_drive_token(self):
        self.start_patch(patch.object(backend, "check_drive_identity"))
        self.start_patch(patch.object(backend.time, "time", self.clock))
        self.start_patch(patch.object(backend.accounts, "drive_token", side_effect=lambda *a, **k: {
            "token": f"t{self.clock.now}", "quota_project": "q", "expires_at": self.clock.now + 3600,
            "folder": "Colab-Persist"}))
        end = self.clock.now + 15 * 3600
        vm = {"expires_at": self.clock.now + 3600}  # Sent by the command that launched the job.
        lapses = []

        def remote_call(operation, *, item, timeout):
            if vm["expires_at"] <= self.clock.now:
                lapses.append(self.clock.now)
            vm["expires_at"] = backend.drive_token(backend.config())["expires_at"]  # What remote_call sends.
            return RUNNING if self.clock.now < end else IDLE
        self.start_patch(patch.object(backend, "remote_call", side_effect=remote_call))
        self.assertEqual(self.run_keeper(), "idle")
        self.assertEqual(lapses, [])
        self.assertGreaterEqual(self.clock.now, end)

    def test_renewal_is_always_due_before_a_sent_token_can_lapse(self):
        self.assertLess(keeper.RENEW_SECONDS + keeper.TICK_SECONDS, backend.TOKEN_MARGIN_SECONDS)

    def test_renews_promptly_when_this_computer_wakes_from_sleep(self):
        renewed_at = []

        def remote_call(operation, *, item, timeout):
            renewed_at.append(self.clock.now)
            return IDLE
        self.start_patch(patch.object(backend, "remote_call", side_effect=remote_call))
        slept = []

        def sleep(seconds):
            # macOS pauses the sleep timer while the lid is closed: the first wait loses three hours of
            # wall-clock time and still needs its full awake duration afterwards.
            slept.append(seconds)
            self.clock.sleep(seconds + (3 * 3600 if len(slept) == 1 else 0))
        woke_at = self.clock.now + 3 * 3600
        self.assertEqual(self.run_keeper(sleep=sleep), "idle")
        self.assertLessEqual(renewed_at[0] - woke_at, keeper.TICK_SECONDS)

    def test_failures_are_retried_within_a_minute_and_never_stop_it(self):
        call = self.statuses(RuntimeError("ssh: Network is unreachable"), subprocess.TimeoutExpired("ssh", 120),
                             RUNNING, IDLE)
        start = self.clock.now
        self.assertEqual(self.run_keeper(), "idle")
        self.assertEqual(call.call_count, 4)
        self.assertEqual(self.clock.now - start, 2 * keeper.RENEW_SECONDS + 2 * keeper.RETRY_SECONDS)
        self.assertTrue(any(line.startswith("Could not renew Drive access yet") for line in self.logs))

    def test_gives_up_after_a_day_without_reaching_the_runtime(self):
        self.start_patch(patch.object(backend, "runtime_info", side_effect=RuntimeError("offline")))
        start = self.clock.now
        self.assertEqual(self.run_keeper(), "unreachable")
        self.assertGreaterEqual(self.clock.now - start, keeper.GIVE_UP_SECONDS)

    def test_stops_when_no_managed_runtime_is_running(self):
        self.start_patch(patch.object(backend, "runtime_info", return_value=None))
        call = self.statuses()
        self.assertEqual(self.run_keeper(), "no_runtime")
        call.assert_not_called()

    def test_stops_when_the_selected_account_changes(self):
        self.start_patch(patch.object(backend, "config", side_effect=[{"expected_email": "one@example.com"},
                                                                      {"expected_email": "two@example.com"}]))
        call = self.statuses()
        self.assertEqual(self.run_keeper(), "account_changed")
        call.assert_not_called()

    def test_a_job_started_while_it_decides_to_stop_keeps_it_running(self):
        def remote_call(operation, *, item, timeout):
            if remote_call.calls == 0:
                (backend.CONFIG_DIR / backend.KEEPER_WANTED).touch()  # ensure_keeper during the check.
                os.utime(backend.CONFIG_DIR / backend.KEEPER_WANTED, (self.clock.now, self.clock.now))
            remote_call.calls += 1
            return IDLE
        remote_call.calls = 0
        self.start_patch(patch.object(backend, "remote_call", side_effect=remote_call))
        self.assertEqual(self.run_keeper(), "idle")
        self.assertEqual(remote_call.calls, 2)

    def test_active_jobs(self):
        self.assertEqual(keeper.active_jobs({"workspaces": {
            "a": {"active": True}, "b": {"active": False, "job": "awaiting_drive"},
            "c": {"active": False, "job": "finished"}, "d": {"active": False}}}), ["a", "b"])
        self.assertEqual(keeper.active_jobs({}), [])


class EnsureKeeperTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.start_patch(patch.dict(os.environ, {"COLAB_PERSIST_NO_KEEPER": ""}))
        self.popen = self.start_patch(patch.object(backend.subprocess, "Popen"))

    def test_starts_one_detached_helper_on_this_computer(self):
        self.assertTrue(backend.ensure_keeper())
        command, kwargs = self.popen.call_args.args[0], self.popen.call_args.kwargs
        self.assertEqual(command, [sys.executable, "-m", "colab_persist.keeper"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue((self.config_dir / backend.KEEPER_WANTED).exists())

    def test_never_starts_a_second_helper(self):
        with open(self.config_dir / backend.KEEPER_LOCK, "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(backend.ensure_keeper())
        self.popen.assert_not_called()

    def test_can_be_turned_off(self):
        with patch.dict(os.environ, {"COLAB_PERSIST_NO_KEEPER": "1"}):
            self.assertFalse(backend.ensure_keeper())
        self.popen.assert_not_called()

    def test_a_failure_to_start_never_fails_the_command(self):
        self.popen.side_effect = OSError("no python")
        self.assertFalse(backend.ensure_keeper())

    def test_a_second_helper_exits_at_once(self):
        with open(self.config_dir / backend.KEEPER_LOCK, "a") as held, \
             patch.object(keeper, "run") as run:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(keeper.main(), 0)
        run.assert_not_called()


class StartedByCommandsTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.managed()
        self.ensure = self.start_patch(patch.object(backend, "ensure_keeper"))
        self.start_patch(patch.object(backend, "upload_source", return_value="train.py"))
        self.script = self.root / "train.py"
        self.script.write_text("print('train')\n")

    def test_run_starts_it_for_detached_and_attached_jobs_but_not_refused_ones(self):
        for detach in (True, False):
            with patch.object(backend, "remote_call", side_effect=[{"restored": None},
                                                                   {"state": "running", "run_id": "r"}]), \
                 patch.object(backend, "monitor", return_value={}):
                backend.run_script(str(self.script), "demo", detach=detach)
        self.assertEqual(self.ensure.call_count, 2)
        with patch.object(backend, "remote_call", side_effect=[{"restored": None},
                                                               {"state": "refused", "error": "busy"}]):
            with self.assertRaises(RuntimeError):
                backend.run_script(str(self.script), "demo")
        self.assertEqual(self.ensure.call_count, 2)

    def test_job_attach_and_status_restart_it_only_while_a_job_runs(self):
        with patch.object(backend, "remote_call", return_value={"state": "finished"}):
            backend.job_status("demo")
        with patch.object(backend, "remote_call", return_value={"state": "finished", "run_id": "r"}), \
             patch.object(backend, "monitor", return_value={}):
            backend.attach("demo")
        with patch.object(backend, "remote_call", return_value=IDLE):
            backend.status()
        self.ensure.assert_not_called()
        with patch.object(backend, "remote_call", return_value={"state": "running"}):
            backend.job_status("demo")
        with patch.object(backend, "remote_call", return_value={"state": "running", "run_id": "r"}), \
             patch.object(backend, "monitor", return_value={}):
            backend.attach("demo")
        with patch.object(backend, "remote_call", return_value=RUNNING):
            backend.status()
        self.assertEqual(self.ensure.call_count, 3)


if __name__ == "__main__":
    unittest.main()
