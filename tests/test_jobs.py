"""Detached managed jobs on the VM: worker, periodic and final checkpoints, launch, job_status."""
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import warnings

from colab_persist import drive_api, remote

import fakes

PROJECT = ["Colab-Persist", "projects", "demo"]
TERMINAL = {"finished", "exited_unsaved", "refused"}


class JobCase(unittest.TestCase):
    def setUp(self):
        fakes.fixed_umask(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.drive = fakes.FakeDrive()
        self.store = remote.Store(self.base / "vm", drive=self.drive)
        self.store.prepare("demo")
        self.workspace = self.store.workspace("demo")
        self.history = []
        original = self.store._write_job

        def recording(project, **values):
            self.history.append(values)
            return original(project, **values)
        self.store._write_job = recording
        # Keep the scripts' caches and tokens inside the temporary directory.
        environment = patch.dict(os.environ, {drive_api.TOKEN_ENV: str(self.base / "token.json")})
        environment.start()
        self.addCleanup(environment.stop)
        for key in ("HF_HOME", "COLAB_DATASET_CACHE", "COLAB_DATASET_ROOT"):
            os.environ.pop(key, None)

    def run_worker(self, script, arguments=(), interval=3600):
        """Run the worker in this process, exactly as the detached subprocess would."""
        (self.workspace / "job.py").write_text(script)
        spec = {"run_id": "run-1", "project": "demo", "script": "job.py", "arguments": list(arguments),
                "interval": interval, "stop_after": True, "folder": "Colab-Persist",
                "snapshot_limit": remote.DEFAULT_SNAPSHOT_LIMIT, "scratch": str(self.store.scratch)}
        remote.write_json(self.store._job_dir("demo") / "job-spec.json", spec)
        self.store._write_job("demo", run_id="run-1", state="starting", stop_after=True)
        with fakes.preserved_sigterm():
            self.store.worker("demo")
        return self.store._read_job("demo")

    def states(self):
        return [values["state"] for values in self.history if "state" in values]

    def manifest_reasons(self):
        reasons = []
        for name in self.drive.listing(PROJECT):
            if name.endswith(".json"):
                reasons.append(self.drive.by_name(name)["appProperties"]["cp_reason"])
        return reasons


class WorkerTests(JobCase):
    def test_periodic_checkpoint_then_final_save_finishes(self):
        job = self.run_worker("import os, pathlib, time\n"
                              "time.sleep(2.5)\n"
                              "pathlib.Path(os.environ['COLAB_OUTPUT_DIR'], 'result.txt').write_text('done')\n",
                              interval=1)
        self.assertEqual(job["state"], "finished", job)
        self.assertEqual(job["exit_code"], 0)
        self.assertIsNone(job["error"])
        self.assertTrue(job.get("last_checkpoint"))
        self.assertIn("periodic", self.manifest_reasons())
        self.assertIn("script_exit", self.manifest_reasons())
        self.assertEqual(job["checkpoint"]["durability"], remote.DURABLE)
        self.assertNotIn("files", job["checkpoint"])
        latest = self.store.latest("demo")["manifest"]
        self.assertEqual(latest["snapshot"], job["checkpoint"]["snapshot"])
        self.assertIn("outputs/result.txt", latest["files"])
        self.assertIn("outputs/last-run.json", latest["files"])
        self.assertEqual(self.states()[:2], ["starting", "running"])
        self.assertEqual(self.states()[-2:], ["saving", "finished"])

    def test_failed_script_still_checkpoints_error_outputs(self):
        job = self.run_worker("print('useful error log'); raise SystemExit(7)\n")
        self.assertEqual(job["state"], "finished")
        self.assertEqual(job["exit_code"], 7)
        latest = self.store.latest("demo")["manifest"]
        self.assertIn("outputs/last-run.json", latest["files"])
        self.assertIn("useful error log", (self.workspace / "outputs" / "run.log").read_text())
        last_run = json.loads((self.workspace / "outputs" / "last-run.json").read_text())
        self.assertEqual(last_run["exit_code"], 7)

    def test_script_receives_arguments_and_workspace_environment(self):
        self.run_worker("import json, os, pathlib, sys\n"
                        "keys = ['COLAB_WORKSPACE', 'COLAB_OUTPUT_DIR', 'COLAB_DATASET_ROOT', 'COLAB_DRIVE_FOLDER']\n"
                        "info = {k: os.environ[k] for k in keys}\n"
                        "info['argv'] = sys.argv[1:]\n"
                        "info['cwd'] = os.getcwd()\n"
                        "pathlib.Path(os.environ['COLAB_OUTPUT_DIR'], 'env.json').write_text(json.dumps(info))\n",
                        arguments=["--steps", "2"])
        info = json.loads((self.workspace / "outputs" / "env.json").read_text())
        self.assertEqual(info["argv"], ["--steps", "2"])
        self.assertEqual(Path(info["COLAB_WORKSPACE"]).resolve(), self.workspace.resolve())
        self.assertEqual(Path(info["cwd"]).resolve(), self.workspace.resolve())
        self.assertEqual(info["COLAB_DATASET_ROOT"], "gdrive://Colab-Persist/datasets")
        self.assertEqual(info["COLAB_DRIVE_FOLDER"], "Colab-Persist")

    def test_data_and_model_caches_are_outside_snapshot(self):
        job = self.run_worker("import os, pathlib\n"
                              "for key in ('COLAB_DATASET_CACHE', 'HF_HOME'):\n"
                              "    p = pathlib.Path(os.environ[key]); p.mkdir(parents=True, exist_ok=True)\n"
                              "    (p / 'cached-weights.bin').write_text('disposable')\n")
        self.assertEqual(job["state"], "finished")
        files = self.store.latest("demo")["manifest"]["files"]
        self.assertFalse(any("cached-weights" in name for name in files))
        self.assertTrue((self.base / "colab-persist-cache" / "huggingface" / "cached-weights.bin").exists())
        self.assertTrue((self.base / "colab-persist-cache" / "datasets" / "cached-weights.bin").exists())

    def test_token_expiry_at_final_save_waits_for_drive_then_succeeds(self):
        self.drive.tokens.expire(times=2)
        with patch.object(remote, "DRIVE_RETRY_SECONDS", 0.05):
            job = self.run_worker("print('finished training')\n")
        self.assertEqual(job["state"], "finished")
        self.assertIsNone(job["error"])
        states = self.states()
        self.assertEqual(states.count("awaiting_drive"), 2)
        self.assertLess(states.index("awaiting_drive"), states.index("finished"))
        waiting = next(values for values in self.history if values.get("state") == "awaiting_drive")
        self.assertIn("expired", waiting["error"])
        self.assertEqual(self.manifest_reasons(), ["script_exit"])

    def test_token_that_never_returns_ends_unsaved_after_the_deadline(self):
        self.drive.tokens.revoked = True
        with patch.object(remote, "FINAL_SAVE_WAIT_SECONDS", -1):
            job = self.run_worker("print('finished training')\n")
        self.assertEqual(job["state"], "exited_unsaved")
        self.assertIn("revoked", job["error"])
        self.assertEqual(self.drive.upload_names(), [])

    def test_quota_exceeded_at_final_save_ends_unsaved_without_retrying(self):
        self.drive.fail_upload(".tar.gz", drive_api.QuotaExceeded(
            "Drive quota exhausted (storageQuotaExceeded): The user's Drive storage quota has been exceeded.",
            403, "storageQuotaExceeded"), times=None)
        with patch.object(remote, "DRIVE_RETRY_SECONDS", 0.05):
            job = self.run_worker("print('finished training')\n")
        self.assertEqual(job["state"], "exited_unsaved")
        self.assertIn("quota", job["error"])
        self.assertNotIn("awaiting_drive", self.states())
        attempts = self.drive.upload_names()
        self.assertEqual(len(attempts), 1)  # One archive attempt; quota errors are not retried.
        self.assertTrue(attempts[0].endswith(".tar.gz"))
        self.assertFalse(any(n.endswith(".json") for n in self.drive.listing(PROJECT)))
        self.assertTrue((self.workspace / "outputs" / "last-run.json").exists())

    def test_periodic_token_expiry_is_recorded_and_the_job_continues(self):
        self.drive.tokens.expire(times=1)
        job = self.run_worker("import time; time.sleep(2.5)\n", interval=1)
        self.assertEqual(job["state"], "finished")
        self.assertTrue(any(values.get("drive_waiting") is True for values in self.history))
        self.assertTrue(any("expired" in error for error in job["periodic_checkpoint_errors"]))

    def test_unexpected_worker_failure_is_recorded_as_terminal(self):
        with patch.object(self.store, "snapshot", side_effect=KeyError("unexpected")):
            with contextlib.suppress(KeyError):
                self.run_worker("print('done')\n")
        self.assertIn(self.store._read_job("demo")["state"], TERMINAL)

    def test_worker_refuses_while_another_job_holds_the_project(self):
        (self.workspace / "job.py").write_text("print('never runs')\n")
        remote.write_json(self.store._job_dir("demo") / "job-spec.json",
                          {"run_id": "run-2", "script": "job.py", "arguments": [], "interval": 60})
        with self.store.project_lock("demo"):
            self.store.worker("demo")
        job = self.store._read_job("demo")
        self.assertEqual(job["state"], "refused")
        self.assertIn("active managed job", job["error"])
        self.assertFalse((self.workspace / "outputs" / "run.log").exists())


class JobStatusTests(JobCase):
    def test_no_job_reports_none(self):
        self.assertEqual(self.store.job_status("demo"), {"state": "none", "project": "demo"})

    def test_offset_and_tail_read_the_log_incrementally(self):
        self.store._write_job("demo", state="running", run_id="r")
        log = self.workspace / "outputs" / "run.log"
        log.write_bytes(b"line one\nline two\n")
        whole = self.store.job_status("demo")
        self.assertEqual(whole["log"], "line one\nline two\n")
        self.assertEqual((whole["offset"], whole["log_bytes"]), (18, 18))
        self.assertEqual(whole["state"], "running")
        with log.open("ab") as stream:
            stream.write(b"line three\n")
        more = self.store.job_status("demo", offset=whole["offset"])
        self.assertEqual(more["log"], "line three\n")
        self.assertEqual(more["offset"], 29)
        self.assertEqual(self.store.job_status("demo", tail=6)["log"], "three\n")
        self.assertEqual(self.store.job_status("demo", offset=999)["log"], "")
        self.assertEqual(self.store.job_status("demo", offset=-5)["log"], "line one\nline two\nline three\n")

    def test_log_is_returned_in_bounded_chunks(self):
        self.store._write_job("demo", state="running")
        (self.workspace / "outputs" / "run.log").write_bytes(b"abcdefghij")
        with patch.object(remote, "LOG_CHUNK_BYTES", 4):
            first = self.store.job_status("demo")
            second = self.store.job_status("demo", offset=first["offset"])
            tail = self.store.job_status("demo", tail=100)
        self.assertEqual((first["log"], first["offset"]), ("abcd", 4))
        self.assertEqual((second["log"], second["offset"]), ("efgh", 8))
        self.assertEqual(tail["log"], "ghij")

    def test_job_without_a_log_yet(self):
        self.store._write_job("demo", state="starting")
        status = self.store.job_status("demo")
        self.assertEqual((status["log"], status["offset"], status["log_bytes"]), ("", 0, 0))


class LaunchTests(JobCase):
    def setUp(self):
        super().setUp()
        (self.workspace / "train.py").write_text("print('training')\n")

    def test_launch_spawns_a_detached_worker_and_returns_after_its_handshake(self):
        spawned = []

        def popen(command, **kwargs):
            spawned.append((command, kwargs))
            spec = json.loads((self.store._job_dir("demo") / "job-spec.json").read_text())
            remote.Store(self.store.scratch, drive=self.drive)._write_job("demo", run_id=spec["run_id"],
                                                                          state="running")
            return MagicMock()
        with patch.object(remote.subprocess, "Popen", side_effect=popen):
            job = self.store.launch("demo", "train.py", ["--steps", "2"], interval=30, run_id="run-xyz",
                                    stop_after=False)
        self.assertEqual((job["state"], job["run_id"]), ("running", "run-xyz"))
        self.assertFalse(job["stop_after"])
        command, kwargs = spawned[0]
        self.assertEqual(command, [sys.executable, "-m", "colab_persist.remote", "--worker", "demo",
                                   "--scratch", str(self.store.scratch)])
        self.assertTrue(kwargs["start_new_session"])
        package_root = str(Path(remote.__file__).resolve().parent.parent)
        self.assertEqual(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], package_root)
        self.assertEqual(str(kwargs["cwd"]), package_root)
        spec = json.loads((self.store._job_dir("demo") / "job-spec.json").read_text())
        self.assertEqual(spec, {"run_id": "run-xyz", "project": "demo", "script": "train.py",
                                "arguments": ["--steps", "2"], "interval": 30, "stop_after": False,
                                "folder": "Colab-Persist", "snapshot_limit": remote.DEFAULT_SNAPSHOT_LIMIT,
                                "scratch": str(self.store.scratch)})

    def test_invalid_requests_never_spawn(self):
        outside = self.base / "outside.py"
        outside.write_text("print('outside')\n")
        (self.workspace / "linked.py").symlink_to(outside)
        (self.workspace / "notes.txt").write_text("text")
        cases = [dict(script="train.py", arguments=[], interval=9),
                 dict(script="train.py", arguments="--steps 2", interval=60),
                 dict(script="train.py", arguments=[2], interval=60),
                 dict(script="../outside.py", arguments=[], interval=60),
                 dict(script=str(outside), arguments=[], interval=60),
                 dict(script="linked.py", arguments=[], interval=60),
                 dict(script="notes.txt", arguments=[], interval=60),
                 dict(script="missing.py", arguments=[], interval=60)]
        with patch.object(remote.subprocess, "Popen") as popen:
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    self.store.launch("demo", case["script"], case["arguments"], interval=case["interval"])
            popen.assert_not_called()
        self.assertFalse((self.store._job_dir("demo") / "job-spec.json").exists())

    def test_launch_refuses_while_a_job_holds_the_project(self):
        with patch.object(remote.subprocess, "Popen") as popen, self.store.project_lock("demo"):
            with self.assertRaisesRegex(RuntimeError, "active managed job"):
                self.store.launch("demo", "train.py", [], interval=60)
        popen.assert_not_called()

    def test_launch_without_drive_fails_before_spawning(self):
        self.drive.tokens = drive_api.TokenSource(path=self.base / "missing-token.json")
        with patch.object(remote.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(drive_api.TokenExpired, "not connected"):
                self.store.launch("demo", "train.py", [], interval=60)
        popen.assert_not_called()

    def test_worker_that_never_reports_is_an_error(self):
        clock = iter(range(0, 1000, 20))
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = lambda: next(clock)
        with patch.object(remote.subprocess, "Popen"), patch.object(remote, "time", fake_time):
            with self.assertRaisesRegex(RuntimeError, "did not start"):
                self.store.launch("demo", "train.py", [], interval=60)
        self.assertEqual(self.store._read_job("demo")["state"], "starting")


class DetachedWorkerEndToEndTests(unittest.TestCase):
    """launch() spawns the real `python -m colab_persist.remote --worker` subprocess.

    The worker reaches a file-backed FakeDrive through a sitecustomize module on
    PYTHONPATH that swaps drive_api.DriveClient; nothing else is altered."""

    def setUp(self):
        fakes.fixed_umask(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def test_launch_runs_the_real_worker_and_checkpoints_to_drive(self):
        state = self.base / "fake-drive"
        boot = self.base / "boot"
        boot.mkdir()
        (boot / "sitecustomize.py").write_text(fakes.SITECUSTOMIZE)
        token = drive_api.write_token_file(self.base / "shm" / "drive-token.json", {
            "token": "fake-access-token", "quota_project": fakes.QUOTA_PROJECT,
            "expires_at": int(time.time()) + 3600, "folder": "Colab-Persist"})
        environment = {"PYTHONPATH": str(boot), "COLAB_PERSIST_FAKE_DRIVE": str(state),
                       "COLAB_PERSIST_FAKES_FILE": str(Path(fakes.__file__).resolve()),
                       drive_api.TOKEN_ENV: str(token)}
        drive = fakes.FakeDrive(state_dir=state)
        store = remote.Store(self.base / "vm", drive=drive)
        store.prepare("demo")
        (store.workspace("demo") / "train.py").write_text(
            "import os, pathlib, sys\n"
            "pathlib.Path(os.environ['COLAB_OUTPUT_DIR'], 'model.txt').write_text(' '.join(sys.argv[1:]))\n"
            "print('training done', flush=True)\n")
        with patch.dict(os.environ, environment), warnings.catch_warnings():
            # launch() deliberately drops the detached worker's Popen handle while it runs.
            warnings.simplefilter("ignore", ResourceWarning)
            job = store.launch("demo", "train.py", ["--steps", "2"], interval=10)
        self.assertIn(job["state"], {"running", "saving", "finished"})
        deadline = time.monotonic() + 60
        status = store.job_status("demo")
        while status["state"] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(0.2)
            status = store.job_status("demo")
        worker_log = store._job_dir("demo") / "worker.log"
        detail = worker_log.read_text() if worker_log.exists() else "(no worker.log)"
        self.assertEqual(status["state"], "finished", detail)
        self.assertEqual(status["exit_code"], 0)
        self.assertIn("training done", status["log"])
        self.assertEqual(status["checkpoint"]["durability"], remote.DURABLE)
        snapshot = status["checkpoint"]["snapshot"]
        self.assertEqual(sorted(drive.listing(PROJECT)), [snapshot + ".json", snapshot + ".tar.gz"])
        replacement = remote.Store(self.base / "vm2", drive=fakes.FakeDrive(state_dir=state))
        replacement.prepare("demo")
        self.assertEqual((replacement.workspace("demo") / "outputs" / "model.txt").read_text(), "--steps 2")

    def test_worker_entry_point_reads_the_job_spec(self):
        """`--worker` builds the Store from job-spec.json (scratch, folder and limit)."""
        scratch = self.base / "vm"
        (scratch / "demo").mkdir(parents=True)
        remote.write_json(scratch / "demo" / "job-spec.json",
                          {"folder": "Custom-Folder", "snapshot_limit": 1234, "scratch": str(scratch)})
        with patch.object(remote, "Store") as store:
            remote._worker_main(["remote.py", "--worker", "demo", "--scratch", str(scratch)])
        store.assert_called_once_with(scratch=str(scratch), folder="Custom-Folder", snapshot_limit=1234)
        store.return_value.worker.assert_called_once_with("demo")
        with self.assertRaises(ValueError):
            remote._worker_main(["remote.py", "--worker", "../escape", "--scratch", str(scratch)])


if __name__ == "__main__":
    unittest.main()
