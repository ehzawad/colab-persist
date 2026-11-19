"""One SSH connection per runtime, and stops that delete only the exact saved generation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import backend, drive_api, remote, runtimes

import fakes
from support import EMAIL, BackendFixture


def receipt_with(*checkpoints, verified=True):
    return {"drive_upload_verified": verified, "drive_flushed": verified, "checkpoints": list(checkpoints),
            "verified_at": "2026-11-20T00:00:00+00:00", "flushed_at": "2026-11-20T00:00:00+00:00"}


class StopTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend()
        self.browser = fakes.make_runtime("browser-notebook-1", fakes.GPU_T4)
        self.api.runtimes.append(self.browser)
        self.runtime = self.managed("cp-cuda-aaa111")
        self.ssh = self.start_patch(patch.object(backend.subprocess, "run"))

    def test_nothing_to_save_stop_deletes_only_the_ledger_runtime(self):
        item = backend.require_runtime()
        with patch.object(backend, "remote_call", return_value={
                "drive_upload_verified": False, "drive_flushed": False, "nothing_to_save": True,
                "checkpoints": []}) as call:
            result = backend._save_and_stop(item, True)
        call.assert_called_once_with("seal", item=item)
        self.assertEqual(self.api.deleted, ["cp-cuda-aaa111"])
        self.assertIn(self.browser, self.api.runtimes)
        self.assertIsNone(self.ledger())
        self.assertFalse((self.config_dir / "last-save.json").exists())
        self.assertTrue(result["stopped"])
        self.assertTrue(result["nothing_to_save"])
        self.assertFalse(result["drive_upload_verified"])
        closing = self.ssh.call_args.args[0]
        self.assertIn("ControlPath=" + str(backend.control_path(item)), closing)
        self.assertEqual(closing[-3:], ["-O", "exit", "root@colab-runtime"])

    def test_a_different_current_runtime_is_never_deleted(self):
        item = backend.require_runtime()

        def replaced_during_save(*_args, **_kwargs):
            runtimes.forget(self.config_dir, EMAIL, "cuda", "cp-cuda-aaa111")
            self.managed("cp-cuda-bbb222")
            return {"nothing_to_save": True, "checkpoints": []}
        with patch.object(backend, "remote_call", side_effect=replaced_during_save):
            with self.assertRaisesRegex(RuntimeError, "Refusing to stop a different VM"):
                backend._save_and_stop(item, True)
        self.assertEqual(self.api.deleted, [])
        self.assertEqual(self.ledger()["runtime_id"], "cp-cuda-bbb222")

    def test_save_refusal_leaves_the_runtime(self):
        item = backend.require_runtime()
        with patch.object(backend, "remote_call", return_value=receipt_with(verified=False)):
            with self.assertRaisesRegex(RuntimeError, "Shutdown refused"):
                backend._save_and_stop(item, True)
        self.assertEqual(self.api.deleted, [])
        self.assertEqual(self.ledger()["runtime_id"], "cp-cuda-aaa111")
        self.ssh.assert_not_called()

    def test_unconfirmed_seal_leaves_the_runtime(self):
        item = backend.require_runtime()
        raw = "Traceback\n" + remote.ERROR_PREFIX + json.dumps({"operation": "seal", "error": "Drive quota exhausted"})
        with patch.object(backend, "checked", return_value=raw.encode()), \
             patch.object(backend, "drive_token", return_value={"token": "t"}):
            with self.assertRaisesRegex(RuntimeError, "Drive quota exhausted.*Runtime was NOT stopped"):
                backend._save_and_stop(item, True)
        self.assertEqual(self.api.deleted, [])

    def test_verified_save_records_a_receipt_and_deletes_the_runtime(self):
        item = backend.require_runtime()
        checkpoint = {"snapshot": "20261120T000000000000Z-aaaaaaaa", "project": "demo", "durability": remote.DURABLE,
                      "files": {"a": {}, "b": {}}, "archive_bytes": 10}
        with patch.object(backend, "remote_call", return_value=receipt_with(checkpoint)):
            result = backend._save_and_stop(item, True)
        self.assertEqual(self.api.deleted, ["cp-cuda-aaa111"])
        self.assertTrue(result["drive_upload_verified"])
        self.assertEqual(result["checkpoints"][0]["file_count"], 2)
        self.assertNotIn("files", result["checkpoints"][0])
        saved = self.config_dir / "last-save.json"
        self.assertEqual(stat.S_IMODE(saved.stat().st_mode), 0o600)
        self.assertEqual(json.loads(saved.read_text())["checkpoints"][0]["file_count"], 2)

    def test_save_without_stop_keeps_the_runtime(self):
        item = backend.require_runtime()
        with patch.object(backend, "remote_call", return_value=receipt_with()):
            result = backend._save_and_stop(item, False)
        self.assertFalse(result["stopped"])
        self.assertEqual(self.api.deleted, [])
        self.assertEqual(self.ledger()["runtime_id"], "cp-cuda-aaa111")

    def test_runtime_deleted_elsewhere_is_forgotten_not_adopted(self):
        self.api.runtimes = [self.browser]
        with self.assertRaisesRegex(RuntimeError, "No managed runtime is active"):
            backend.require_runtime()
        self.assertIsNone(self.ledger())
        self.assertEqual(self.api.deleted, [])

    def test_safe_stop_takes_the_operation_lock(self):
        with backend.operation_lock():
            with self.assertRaisesRegex(RuntimeError, "Another persistence operation is active"):
                backend.save_and_stop(stop=True)
        self.assertEqual(self.api.deleted, [])


class RemoteCallTests(BackendFixture, unittest.TestCase):
    def setUp(self):
        self.use_backend(snapshot_limit_bytes=1234)
        self.runtime = self.managed()
        self.identity = self.start_patch(patch.object(backend, "check_drive_identity"))

    def test_each_minted_token_is_bound_to_the_selected_account(self):
        token = {"token": "t", "quota_project": "q", "expires_at": 4102444800, "folder": "Colab-Persist"}
        with patch.object(backend.accounts, "drive_token", return_value=token):
            backend.drive_token(backend.config())
            backend.drive_token(backend.config())
        self.identity.assert_called_once()
        self.identity.side_effect = RuntimeError("Drive account mismatch")
        backend._TOKENS.clear()
        with patch.object(backend.accounts, "drive_token", return_value=token):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                backend.drive_token(backend.config())
        self.assertEqual(backend._TOKENS, {})

    def test_request_carries_only_a_short_lived_drive_token(self):
        token = {"token": "fake-access-token", "quota_project": fakes.QUOTA_PROJECT, "expires_at": 4102444800,
                 "folder": "Colab-Persist"}
        sent = []

        def checked(arguments, *, data=None, timeout=None):
            sent.append((arguments, json.loads(data), timeout))
            return (remote.RESULT_PREFIX + json.dumps({"ok": True})).encode()
        with patch.object(backend.accounts, "drive_token", return_value=token) as mint, \
             patch.object(backend, "checked", side_effect=checked):
            self.assertEqual(backend.remote_call("status", item=self.runtime, timeout=42), {"ok": True})
            backend.remote_call("status", item=self.runtime)
        arguments, request, timeout = sent[0]
        self.assertEqual(arguments[-1], backend.ENGINE_COMMAND)
        self.assertEqual(request, {"operation": "status", "folder": "Colab-Persist", "snapshot_limit": 1234,
                                   "drive": token})
        self.assertEqual(timeout, 42)
        mint.assert_called_once()  # Cached until it nears expiry.

    def test_drive_token_is_refreshed_near_expiry(self):
        soon = {"token": "old", "quota_project": "q", "expires_at": 0, "folder": "Colab-Persist"}
        later = {"token": "new", "quota_project": "q", "expires_at": 4102444800, "folder": "Colab-Persist"}
        with patch.object(backend.accounts, "drive_token", side_effect=[soon, later]) as mint:
            cfg = backend.config()
            self.assertEqual(backend.drive_token(cfg)["token"], "old")
            self.assertEqual(backend.drive_token(cfg)["token"], "new")
            self.assertEqual(backend.drive_token(cfg)["token"], "new")
        self.assertEqual(mint.call_count, 2)
        self.assertEqual(mint.call_args.kwargs, {"full_drive": False})


class ParseResultTests(unittest.TestCase):
    def test_engine_reports_why_an_operation_failed(self):
        output = io.StringIO()
        real_store = remote.Store
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            drive = fakes.FakeDrive(tokens=drive_api.TokenSource(path=Path(directory) / "missing-token.json"))
            with patch.object(remote, "Store", side_effect=lambda **kw: real_store(
                    scratch=Path(directory) / "vm", drive=drive, **kw)):
                with self.assertRaisesRegex(drive_api.TokenExpired, "Drive is not connected"):
                    remote.dispatch({"operation": "prepare", "project": "demo"})
        line = output.getvalue().strip().splitlines()[-1]
        self.assertTrue(line.startswith(remote.ERROR_PREFIX))
        detail = json.loads(line[len(remote.ERROR_PREFIX):])
        self.assertEqual((detail["operation"], detail["type"]), ("prepare", "TokenExpired"))
        self.assertIn("Drive is not connected", detail["error"])

    def test_failure_names_the_operation_and_its_reason(self):
        raw = "Traceback...\n" + remote.ERROR_PREFIX + json.dumps({"operation": "status", "error": "disk full"})
        with self.assertRaises(RuntimeError) as status_error:
            backend.parse_result(raw, "status")
        self.assertIn("'status'", str(status_error.exception))
        self.assertIn("disk full", str(status_error.exception))
        self.assertNotIn("NOT stopped", str(status_error.exception))
        with self.assertRaisesRegex(RuntimeError, "Runtime was NOT stopped"):
            backend.parse_result(raw, "seal")
        with self.assertRaisesRegex(RuntimeError, "returned no confirmed result"):
            backend.parse_result("garbage\n" + remote.ERROR_PREFIX + "{not json", "status")

    def test_results_still_parse(self):
        raw = "noise\n" + remote.RESULT_PREFIX + json.dumps({"drive_connected": True})
        self.assertEqual(backend.parse_result(raw, "status"), {"drive_connected": True})


class SharedSSHTests(unittest.TestCase):
    def setUp(self):
        self.first = fakes.make_runtime("cp-cuda-aaa111")
        self.second = fakes.make_runtime("cp-cuda-bbb222")
        config = patch.object(backend, "config", return_value={"ssh_identity": "~/.ssh/id_test"})
        config.start()
        self.addCleanup(config.stop)

    def test_control_path_is_hashed_short_and_per_runtime(self):
        path = backend.control_path(self.first)
        digest = hashlib.sha256(b"cp-cuda-aaa111").hexdigest()[:16]
        self.assertEqual(path, backend.SSH_CONTROL_DIR / ("cm-cp-" + digest))
        self.assertNotEqual(path, backend.control_path(self.second))
        longest = fakes.make_runtime("c" + "x" * 62)
        self.assertEqual(len(backend.control_path(longest).name), len(path.name))
        # OpenSSH appends a 17-character temporary suffix while binding; macOS allows 104 bytes.
        self.assertLess(len(str(backend.control_path(longest)).encode()) + 17, 104)
        with patch.object(backend, "SSH_CONTROL_DIR", Path("/home/a-fairly-long-user-name-for-ci/.ssh")):
            self.assertLess(len(str(backend.control_path(longest)).encode()) + 17, 104)

    def test_commands_share_one_connection_per_runtime(self):
        first = backend.ssh_command(self.first)
        second = backend.ssh_command(self.second)
        self.assertIn("ControlMaster=auto", first)
        self.assertIn("ControlPersist=30", first)
        self.assertIn("HostKeyAlias=colab-cp-cuda-aaa111", first)
        self.assertIn("BatchMode=yes", first)
        self.assertIn("ForwardAgent=no", first)
        paths = [arg for arg in first + second if arg.startswith("ControlPath=")]
        self.assertEqual(paths, ["ControlPath=" + str(backend.control_path(self.first)),
                                 "ControlPath=" + str(backend.control_path(self.second))])
        proxy = next(arg for arg in first if arg.startswith("ProxyCommand="))
        self.assertTrue(proxy.endswith("-m colab_persist.proxy --runtime cp-cuda-aaa111"))
        self.assertEqual(first[-1], "root@colab-runtime")

    def test_alias_uses_runtime_pinning_and_the_same_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ssh" / "colab-cuda.conf"
            with patch.object(backend, "config", return_value={"ssh_alias_config": str(target)}):
                backend.write_ssh_alias(self.first)
            text = target.read_text()
            mode = stat.S_IMODE(target.stat().st_mode)
        self.assertEqual(mode, 0o600)
        self.assertIn("Host colab-cuda", text)
        self.assertIn("HostKeyAlias colab-cp-cuda-aaa111", text)
        self.assertIn("--runtime cp-cuda-aaa111", text)
        self.assertNotIn("--endpoint", text)
        self.assertIn("ControlMaster auto", text)
        self.assertIn(f'ControlPath "{backend.control_path(self.first)}"', text)

    def test_alias_is_optional(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(backend, "config", return_value={}):
            backend.write_ssh_alias(self.first)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_busy_ssh_slot_is_explained(self):
        busy = SimpleNamespace(returncode=255, stdout=b"",
                               stderr=b"The runtime refused SSH: Already-active SSH session (HTTP 429)")
        with patch.object(backend.subprocess, "run", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "one SSH session per runtime.*ssh -O exit colab-cuda"):
                backend.checked(["ssh", "colab-runtime", "true"])

    def test_other_command_failures_include_output(self):
        failed = SimpleNamespace(returncode=2, stdout=b"partial", stderr=b"permission denied")
        with patch.object(backend.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, r"Command failed \(2\): permission denied"):
                backend.checked(["ssh", "colab-runtime", "true"])


if __name__ == "__main__":
    unittest.main()
