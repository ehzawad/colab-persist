"""Local side of detached jobs: run_script, job_status, attach and monitor."""
import io
import unittest
from unittest.mock import patch

from colab_persist import backend, remote

from support import BackendFixture


class MonitorTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.runtime = self.managed()
        self.start_patch(patch.object(backend, "POLL_SECONDS", 0))
        self.save = self.start_patch(patch.object(backend, "_save_and_stop",
                                                  return_value={"stopped": True, "drive_upload_verified": True}))
        self.log = io.StringIO()

    def statuses(self, *items):
        # A finished job is re-read under the lock before the final save; repeat the last state.
        final = [items[-1]] if items and isinstance(items[-1], dict) and items[-1].get("state") == "finished" else []
        return self.start_patch(patch.object(backend, "remote_call", side_effect=[*items, *final]))

    def test_follows_the_log_then_saves_and_stops(self):
        call = self.statuses({"state": "running", "log": "epoch 1\n", "offset": 8, "stop_after": True},
                             {"state": "saving", "log": "", "offset": 8, "stop_after": True},
                             {"state": "finished", "log": "done\n", "offset": 13, "exit_code": 0, "stop_after": True,
                              "checkpoint": {"snapshot": "s1"}, "periodic_checkpoint_errors": ["transient"]})
        result = backend.monitor("demo", item=self.runtime, log=self.log)
        self.assertEqual(self.log.getvalue(), "epoch 1\ndone\n")
        self.assertEqual([c.kwargs["offset"] for c in call.call_args_list[:3]], [0, 8, 8])
        self.assertEqual(call.call_args_list[3].kwargs["tail"], 0)  # Final recheck under the lock.
        self.save.assert_called_once_with(self.runtime, True)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["checkpoint"], {"snapshot": "s1", "durability": remote.DURABLE})
        self.assertEqual(result["log_tail"], "epoch 1\ndone\n")
        self.assertEqual(result["periodic_checkpoint_errors"], ["transient"])

    def test_stop_after_from_the_job_or_the_caller(self):
        self.statuses({"state": "finished", "stop_after": False, "checkpoint": {}})
        backend.monitor("demo", item=self.runtime, log=self.log)
        self.save.assert_called_with(self.runtime, False)
        self.statuses({"state": "finished", "stop_after": True, "checkpoint": {}})
        backend.monitor("demo", item=self.runtime, log=self.log, stop_after=False)
        self.save.assert_called_with(self.runtime, False)

    def test_unsaved_or_refused_jobs_leave_the_runtime_running(self):
        self.statuses({"state": "exited_unsaved", "error": "Drive quota exhausted"})
        with self.assertRaisesRegex(RuntimeError, "final checkpoint was not saved: Drive quota exhausted.*left running"):
            backend.monitor("demo", item=self.runtime, log=self.log)
        self.statuses({"state": "refused", "error": "Project demo has an active managed job"})
        with self.assertRaisesRegex(RuntimeError, "active managed job"):
            backend.monitor("demo", item=self.runtime, log=self.log)
        self.save.assert_not_called()

    def test_lost_contact_checks_whether_the_runtime_still_exists(self):
        self.statuses(RuntimeError("ssh: connection reset"), {"state": "finished", "checkpoint": {}})
        with patch.object(backend.time, "sleep") as sleep:
            backend.monitor("demo", item=self.runtime, log=self.log)
        sleep.assert_any_call(5)
        self.api.runtimes = []
        self.statuses(RuntimeError("ssh: connection reset"))
        with patch.object(backend.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "runtime disappeared"):
                backend.monitor("demo", item=self.runtime, log=self.log)

    def test_attach_starts_from_the_current_offset(self):
        call = self.statuses({"state": "running", "log": "old\n", "offset": 4},
                             {"state": "finished", "log": "new\n", "offset": 8, "checkpoint": {}})
        backend.attach("demo", log=self.log)
        self.assertEqual(call.call_args_list[0].kwargs["tail"], 4000)
        self.assertEqual(call.call_args_list[1].kwargs["offset"], 4)
        self.assertEqual(self.log.getvalue(), "old\nnew\n")

    def test_a_job_replaced_mid_follow_is_never_finalized(self):
        self.statuses({"state": "running", "run_id": "A", "log": "", "offset": 0, "stop_after": False},
                      {"state": "running", "run_id": "B", "log": "", "offset": 0, "stop_after": True})
        with self.assertRaisesRegex(RuntimeError, "different job replaced"):
            backend.monitor("demo", item=self.runtime, log=self.log, run_id="A")
        self.save.assert_not_called()

    def test_finalization_rechecks_the_run_under_the_lock(self):
        self.start_patch(patch.object(backend, "remote_call", side_effect=[
            {"state": "finished", "run_id": "A", "checkpoint": {}, "stop_after": True},
            {"state": "running", "run_id": "B"}]))
        with self.assertRaisesRegex(RuntimeError, "another job started.*nothing was stopped"):
            backend.monitor("demo", item=self.runtime, log=self.log, run_id="A")
        self.save.assert_not_called()

    def test_attach_without_a_job(self):
        self.statuses({"state": "none", "project": "demo"})
        with self.assertRaisesRegex(RuntimeError, "No managed job has run in project demo"):
            backend.attach("demo")
        self.save.assert_not_called()

    def test_job_status_is_one_remote_look(self):
        call = self.statuses({"state": "running"})
        self.assertEqual(backend.job_status("demo"), {"state": "running"})
        self.assertEqual(call.call_args.args, ("job_status",))
        self.assertEqual(call.call_args.kwargs["project"], "demo")
        with self.assertRaises(ValueError):
            backend.job_status("../escape")


class RunScriptTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.runtime = self.managed()
        self.script = self.root / "train.py"
        self.script.write_text("print('train')\n")
        self.upload = self.start_patch(patch.object(backend, "upload_source", return_value="train.py"))

    def test_detached_run_returns_after_launch(self):
        with patch.object(backend, "remote_call", side_effect=[{"restored": None},
                                                               {"state": "running", "run_id": "r1"}]) as call, \
             patch.object(backend, "monitor") as monitor:
            result = backend.run_script(str(self.script), "demo", ["--steps", "2"], checkpoint_seconds=30,
                                        stop_after=False, detach=True)
        monitor.assert_not_called()
        self.assertEqual(result, {"detached": True, "project": "demo", "run_id": "r1", "state": "running",
                                  "restored": None, "next": "colab-persist attach --project demo"})
        launch = call.call_args_list[1]
        self.assertEqual(launch.args, ("launch",))
        self.assertEqual(launch.kwargs["script"], "train.py")
        self.assertEqual(launch.kwargs["arguments"], ["--steps", "2"])
        self.assertEqual(launch.kwargs["interval"], 30)
        self.assertFalse(launch.kwargs["stop_after"])

    def test_attached_run_monitors_and_reports_the_restore(self):
        with patch.object(backend, "remote_call", side_effect=[{"restored": {"snapshot": "s0"}},
                                                               {"state": "running", "run_id": "r1"}]), \
             patch.object(backend, "monitor", return_value={"exit_code": 0}) as monitor:
            result = backend.run_script(str(self.script), "demo")
        monitor.assert_called_once()
        self.assertEqual(result, {"exit_code": 0, "restored": {"snapshot": "s0"}})

    def test_refused_launch_is_an_error(self):
        with patch.object(backend, "remote_call", side_effect=[{"restored": None},
                                                               {"state": "refused", "error": "busy"}]):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                backend.run_script(str(self.script), "demo")

    def test_run_requires_a_managed_runtime(self):
        self.api.runtimes = []
        with self.assertRaisesRegex(RuntimeError, "No managed runtime is active"):
            backend.run_script(str(self.script), "demo")
        self.upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
