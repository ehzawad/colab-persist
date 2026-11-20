"""Regressions for defects found in the independent 0.4.0 implementation review."""
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from colab_persist import accounts, backend, colab_api, drive_api, remote, runtimes

import fakes
from fakes import FakeResponse, ScriptedOpener, drive_error_body, json_body
from support import EMAIL, BackendFixture


class StoreBase(unittest.TestCase):
    def setUp(self):
        fakes.fixed_umask(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.drive = fakes.FakeDrive()
        self.store = remote.Store(self.base / "vm", drive=self.drive)
        self.store.prepare("demo")
        self.workspace = self.store.workspace("demo")


class RestoreSafetyTests(StoreBase):
    def test_a_manifest_naming_paths_outside_the_workspace_never_changes_their_mode(self):
        (self.workspace / "data").write_text("x")
        saved = self.store.snapshot("demo")
        victim = self.base / "victim.txt"
        victim.write_text("keep my mode")
        os.chmod(victim, 0o600)
        fresh = remote.Store(self.base / "vm-other", drive=self.drive)
        # Restore extracts into <workspace parent>/restore-XXXX/workspace; aim the escape at the victim from there.
        staging = fresh.workspace("demo").parent / "restore-x" / "workspace"
        escape = os.path.relpath(victim, staging)
        self.assertTrue(escape.startswith(".."))
        tampered = dict(saved)
        tampered["files"] = {**saved["files"], escape: {"size": 1, "sha256": "0" * 64, "mode": 0o777}}
        tampered = {k: v for k, v in tampered.items() if k not in ("reused", "durability", "manifest_file_id",
                                                                    "archive_file_id")}
        self.drive.corrupt(saved["manifest_file_id"], remote.manifest_bytes(tampered))
        with self.assertRaises((ValueError, RuntimeError)):
            fresh.restore("demo")
        self.assertEqual(os.stat(victim).st_mode & 0o777, 0o600)


class FinalSaveTests(StoreBase):
    def test_a_permanent_drive_error_ends_the_job_instead_of_waiting_a_day(self):
        (self.workspace / "job.py").write_text("print('done')\n")
        spec = {"run_id": "r", "script": "job.py", "arguments": [], "interval": 600}
        permanent = drive_api.DriveError("Drive POST failed (400 badRequest): no", 400, "badRequest")
        with patch.object(self.store, "snapshot", side_effect=permanent), \
             patch.object(remote, "DRIVE_RETRY_SECONDS", 0), fakes.preserved_sigterm():
            self.store._work("demo", spec)
        self.assertEqual(self.store._read_job("demo")["state"], "exited_unsaved")

    def test_a_live_pid_without_the_lock_is_lost(self):
        self.store._write_job("demo", fresh=True, run_id="r", state="running", pid=os.getpid())
        self.assertEqual(self.store.job_status("demo")["state"], "lost")


class DriveClientTests(unittest.TestCase):
    def client(self, *responses, retries=6):
        self.opener = ScriptedOpener(*responses)
        self.addCleanup(self.opener.close)
        return drive_api.DriveClient(drive_api.TokenSource("t", fakes.QUOTA_PROJECT), opener=self.opener,
                                     sleep=lambda _s: None, retries=retries)

    def test_a_401_on_the_last_retry_is_still_token_expired(self):
        drive = self.client((401, {}, drive_error_body(401, "authError")),
                            (401, {}, drive_error_body(401, "authError")), retries=0)
        with self.assertRaises(drive_api.TokenExpired):
            drive.about()

    def test_a_download_complete_before_the_drop_is_not_requested_again(self):
        data = b"0123456789"
        drive = self.client(FakeResponse(200, chunks=[data, ConnectionResetError("reset at EOF")]))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "file"
            drive.download("id-1", target, expected_size=10)
            self.assertEqual(target.read_bytes(), data)
        self.assertEqual(len(self.opener.requests), 1)


class MonitorResilienceTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.runtime = self.managed()
        self.start_patch(patch.object(backend, "POLL_SECONDS", 0))
        self.start_patch(patch.object(backend.time, "sleep"))
        self.save = self.start_patch(patch.object(backend, "_save_and_stop", return_value={"stopped": True}))
        self.log = io.StringIO()

    def test_an_ssh_timeout_is_retried(self):
        finished = {"state": "finished", "run_id": "r", "checkpoint": {}}
        self.start_patch(patch.object(backend, "remote_call", side_effect=[
            subprocess.TimeoutExpired("ssh", 120), finished, finished]))
        result = backend.monitor("demo", item=self.runtime, log=self.log, run_id="r")
        self.assertIn("checkpoint", result)

    def test_a_replaced_runtime_stops_the_follow_at_once(self):
        self.start_patch(patch.object(backend, "remote_call", side_effect=RuntimeError("ssh: gone")))
        self.start_patch(patch.object(backend, "runtime_info", return_value=fakes.make_runtime("cp-cuda-other1")))
        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            backend.monitor("demo", item=self.runtime, log=self.log, run_id="r")

    def test_the_whole_log_is_drained_and_drive_waits_are_shown(self):
        self.start_patch(patch.object(backend, "remote_call", side_effect=[
            {"state": "awaiting_drive", "run_id": "r", "error": "token expired", "log": "", "offset": 0,
             "log_bytes": 0},
            {"state": "finished", "run_id": "r", "log": "part one ", "offset": 9, "log_bytes": 17, "checkpoint": {}},
            {"state": "finished", "run_id": "r", "log": "part two", "offset": 17, "log_bytes": 17, "checkpoint": {}},
            {"state": "finished", "run_id": "r", "checkpoint": {}}]))
        result = backend.monitor("demo", item=self.runtime, log=self.log, run_id="r")
        self.assertIn("Waiting to save to Drive: token expired", self.log.getvalue())
        self.assertTrue(result["log_tail"].endswith("part one part two"))


class RemoteCallErrorTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.runtime = self.managed()
        self.start_patch(patch.object(backend, "drive_token", return_value={"token": "t"}))

    def test_the_engines_reason_is_reported_and_seal_says_the_runtime_kept_running(self):
        failure = RuntimeError("Command failed (1): " + remote.ERROR_PREFIX
                               + json.dumps({"operation": "seal", "error": "Drive quota exhausted"}))
        with patch.object(backend, "checked", side_effect=failure), \
             patch.object(backend, "install_engine") as install:
            with self.assertRaisesRegex(RuntimeError, "Drive quota exhausted.*NOT stopped"):
                backend.remote_call("seal", item=self.runtime)
        install.assert_not_called()

    def test_an_unrelated_missing_file_never_reinstalls_or_repeats_the_operation(self):
        with patch.object(backend, "checked", side_effect=RuntimeError("cat: /content/x: No such file or directory")) \
                as checked, patch.object(backend, "install_engine") as install:
            with self.assertRaises(RuntimeError):
                backend.remote_call("status", item=self.runtime)
        install.assert_not_called()
        self.assertEqual(checked.call_count, 1)


class TokenCacheTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.start_patch(patch.object(backend, "check_drive_identity"))

    def test_a_new_project_or_reauth_elsewhere_mints_a_new_token(self):
        token = {"token": "t", "quota_project": "q", "expires_at": 4102444800, "folder": "Colab-Persist"}
        with patch.object(backend.accounts, "drive_token", return_value=token) as mint:
            backend.drive_token(backend.config())
            accounts.save_meta(self.config_dir, EMAIL, quota_project="another-project")
            backend.drive_token(backend.config())
            credentials = accounts.credentials_path(backend.config())
            os.utime(credentials, ns=(1, 1))
            backend.drive_token(backend.config())
        self.assertEqual(mint.call_count, 3)


class AccountSwitchTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()

    def test_switching_refuses_while_the_old_account_has_a_pending_creation(self):
        runtimes.begin(self.config_dir, EMAIL, "cuda", fakes.GPU_L4)
        other = accounts.account_dir(self.config_dir, "second@example.com") / "credentials.json"
        other.write_text(json.dumps({"email": "second@example.com"}))
        with patch.object(backend, "verify_account", return_value={"account_verified": True}):
            with self.assertRaisesRegex(RuntimeError, "creation still pending"):
                backend.login("second@example.com")


class SnapshotListingTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()

    def test_a_broken_marker_is_skipped_and_duplicates_are_listed_once(self):
        project = ["Colab-Persist", "projects", "demo"]
        manifest = {"schema": 1, "project": "demo", "snapshot": "20260101T000000000000Z-aaaaaaaa",
                    "created_at": "2026-01-01T00:00:00+00:00", "reason": "manual", "files": {},
                    "archive_sha256": "", "archive_bytes": 0}
        archive = b"archive bytes"
        manifest["archive_sha256"] = __import__("hashlib").sha256(archive).hexdigest()
        manifest["archive_bytes"] = len(archive)
        self.drive.put(project, manifest["snapshot"] + ".tar.gz", archive)
        for _ in range(2):  # A retried manifest upload.
            self.drive.put(project, manifest["snapshot"] + ".json", remote.manifest_bytes(manifest),
                           app_properties=remote.summary_properties(manifest))
        self.drive.put(project, "20260202T000000000000Z-bbbbbbbb.json", b"{broken")
        listed = backend.snapshots("demo")
        self.assertEqual([item["snapshot"] for item in listed], [manifest["snapshot"]])


class LegacyNamesTests(StoreBase):
    def test_folders_that_are_not_project_names_are_ignored(self):
        legacy = self.base / "legacy"
        (legacy / "MyDrive" / "Colab-CUDA" / "projects" / ".ipynb_checkpoints").mkdir(parents=True)
        (legacy / "MyDrive" / "Colab-CUDA" / "projects" / "my project").mkdir()
        store = remote.Store(self.base / "vm-legacy", drive=self.drive, legacy_mount=legacy)
        with patch("colab_persist.remote.os.path.ismount", side_effect=lambda p: Path(p) == legacy):
            report = store.import_legacy()
        self.assertEqual(sorted(report["ignored_folders"]), [".ipynb_checkpoints", "my project"])
        self.assertEqual(report["imported"], [])


class PreviousRunLogTests(StoreBase):
    """Found in live verification: run.log travels with checkpoints, so a new run must not replay it."""

    def launch_with_output(self, new_output):
        (self.workspace / "train.py").write_text("print('training')\n")

        def popen(command, **kwargs):
            spec = json.loads((self.store._job_dir("demo") / "job-spec.json").read_text())
            with (self.workspace / "outputs" / "run.log").open("a") as log:
                log.write(new_output)
            remote.Store(self.store.scratch, drive=self.drive)._write_job("demo", run_id=spec["run_id"],
                                                                          state="running")
            return MagicMock()
        with patch.object(remote.subprocess, "Popen", side_effect=popen):
            return self.store.launch("demo", "train.py", [], interval=30)

    def test_a_new_run_reports_only_its_own_log(self):
        (self.workspace / "outputs").mkdir(exist_ok=True)
        (self.workspace / "outputs" / "run.log").write_text("old run line\n" * 1000)
        job = self.launch_with_output("new run line\n")
        followed = self.store.job_status("demo", offset=job["log_start"])
        self.assertEqual(followed["log"], "new run line\n")
        self.assertEqual(self.store.job_status("demo", tail=4000)["log"], "new run line\n")

    def test_a_first_run_starts_at_the_beginning(self):
        job = self.launch_with_output("first line\n")
        self.assertEqual(job["log_start"], 0)
        self.assertEqual(self.store.job_status("demo", tail=4000)["log"], "first line\n")


class RunFollowOffsetTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.managed()
        self.start_patch(patch.object(backend, "upload_source", return_value="train.py"))

    def test_an_attached_run_follows_from_where_its_own_log_starts(self):
        script = self.root / "train.py"
        script.write_text("print('train')\n")
        with patch.object(backend, "remote_call", side_effect=[{"restored": {"snapshot": "s0"}},
                                                               {"state": "running", "run_id": "r1",
                                                                "log_start": 13000}]), \
             patch.object(backend, "monitor", return_value={"exit_code": 0}) as monitor:
            backend.run_script(str(script), "demo")
        self.assertEqual(monitor.call_args.kwargs["offset"], 13000)


if __name__ == "__main__":
    unittest.main()
