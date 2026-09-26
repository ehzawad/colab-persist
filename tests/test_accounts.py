import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import accounts, backend, client, session_state


class FakeAuthorizedSession:
    def __init__(self, credentials):
        self.credentials = credentials

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get(self, url, timeout):
        if self.credentials.get("expired"):
            raise RuntimeError("fake expired credentials")
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"email": self.credentials["email"]})


class AccountTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "persist"
        self.config_dir.mkdir()
        self.first = "first@example.com"
        self.second = "second@example.com"
        self.first_path = self.cache(self.first, marker="original-first")
        self.cfg = {"expected_email": self.first, "credentials_file": str(self.first_path),
                    "session": "training", "gpu": "A100", "drive_folder": "Training",
                    "ssh_identity": str(self.root / "fake-key")}
        self.config_path = self.config_dir / "config.json"
        self.config_path.write_text(json.dumps(self.cfg))
        self.loaded_paths = []
        self.assignment_calls = []
        self.assignment_results = {}
        self.login_calls = []
        self.login_exit_code = 0
        self.login_email = None
        self.login_colab_denied = False

        patches = [
            patch.object(backend, "CONFIG_DIR", self.config_dir),
            patch.object(accounts.google.auth, "load_credentials_from_file", side_effect=self.load),
            patch.object(accounts, "_get_adc_credentials", side_effect=AssertionError("Unexpected global ADC read")),
            patch.object(accounts, "AuthorizedSession", FakeAuthorizedSession),
            patch.object(backend, "AuthorizedSession", FakeAuthorizedSession),
            patch.object(backend, "Client", side_effect=self.make_client),
            patch.object(backend.subprocess, "run", side_effect=self.gcloud),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def cache(self, email, **data):
        path = session_state.session_path(self.config_dir, email).parent / "credentials.json"
        path.write_text(json.dumps({"email": email, **data}))
        path.chmod(0o600)
        return path

    def load(self, filename, scopes):
        self.loaded_paths.append(Path(filename))
        self.assertEqual(scopes, list(accounts.SCOPES))
        return json.loads(Path(filename).read_text()), None

    def make_client(self, _environment, authorized):
        def assignments():
            email = authorized.credentials["email"]
            self.assignment_calls.append(email)
            if authorized.credentials.get("colab_denied"):
                raise RuntimeError("fake Colab permission denied")
            results = self.assignment_results.get(email)
            return results.pop(0) if results else []
        return SimpleNamespace(list_assignments=assignments)

    def gcloud(self, command, *, env):
        self.assertEqual(command[:4], ["gcloud", "auth", "application-default", "login"])
        self.login_calls.append((command, env.copy()))
        staged = Path(env["CLOUDSDK_CONFIG"]) / "application_default_credentials.json"
        staged.write_text(json.dumps({"email": self.login_email or command[4],
                                      "marker": "new-browser-login",
                                      "colab_denied": self.login_colab_denied}))
        return SimpleNamespace(returncode=self.login_exit_code)

    def before(self, *extra_paths):
        return {path: path.read_bytes() for path in (self.config_path, self.first_path, *extra_paths)}

    def assert_unchanged(self, previous):
        for path, content in previous.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_switch_first_second_first_uses_cached_private_credentials(self):
        second_path = self.cache(self.second, marker="original-second")
        first_bytes, second_bytes = self.first_path.read_bytes(), second_path.read_bytes()
        result = backend.login(self.second)
        self.assertTrue(result["switched"])
        self.assertEqual(backend.config()["expected_email"], self.second)
        self.assertEqual(backend.config()["credentials_file"], str(second_path))
        result = backend.login(self.first)
        self.assertTrue(result["switched"])
        self.assertEqual(backend.config(), self.cfg)
        self.assertEqual(self.login_calls, [])
        self.assertEqual(self.first_path.read_bytes(), first_bytes)
        self.assertEqual(second_path.read_bytes(), second_bytes)
        self.assertIn(self.first_path, self.loaded_paths)
        self.assertIn(second_path, self.loaded_paths)
        self.assertEqual(stat.S_IMODE(self.config_path.stat().st_mode), 0o600)

    def test_browser_login_isolated_from_shell_and_global_gcloud(self):
        original_adc = self.root / "unrelated-adc.json"
        original_sdk = self.root / "unrelated-gcloud"
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": str(original_adc),
                                    "CLOUDSDK_CONFIG": str(original_sdk)}):
            result = backend.login(self.second, no_launch_browser=True)
            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], str(original_adc))
            self.assertEqual(os.environ["CLOUDSDK_CONFIG"], str(original_sdk))
        self.assertFalse(result["global_gcloud_credentials_changed"])
        command, environment = self.login_calls[0]
        staging_dir = Path(environment["CLOUDSDK_CONFIG"])
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", environment)
        self.assertEqual(staging_dir.parent,
                         session_state.session_path(self.config_dir, self.second).parent)
        self.assertNotEqual(staging_dir, original_sdk)
        self.assertFalse(staging_dir.exists())
        self.assertFalse(original_adc.exists())
        self.assertFalse(original_sdk.exists())
        self.assertIn("--no-launch-browser", command)
        self.assertIn("--disable-quota-project", command)
        self.assertIn("--scopes=" + ",".join(accounts.SCOPES), command)
        selected = Path(backend.config()["credentials_file"])
        self.assertEqual(json.loads(selected.read_text())["email"], self.second)
        self.assertEqual(stat.S_IMODE(selected.stat().st_mode), 0o600)

    def test_private_credentials_override_unrelated_shell_credentials(self):
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/unrelated/account.json"}):
            environment = accounts.environment(self.cfg)
            verified = accounts.verified_credentials(self.cfg)
            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "/unrelated/account.json")
        self.assertEqual(environment["GOOGLE_APPLICATION_CREDENTIALS"], str(self.first_path))
        self.assertEqual(verified["email"], self.first)
        self.assertEqual(self.loaded_paths, [self.first_path])

    def test_wrong_google_email_fails_identity_verification(self):
        cfg = {**self.cfg, "expected_email": self.second}
        with self.assertRaisesRegex(RuntimeError, "Google account mismatch"):
            accounts.verified_credentials(cfg)
        self.assertEqual(self.assignment_calls, [])

    def test_missing_private_credentials_do_not_fall_back_to_shell(self):
        cfg = {**self.cfg, "credentials_file": str(self.root / "missing.json")}
        with self.assertRaisesRegex(RuntimeError, "Saved account credentials are missing"):
            accounts.environment(cfg)
        with self.assertRaisesRegex(RuntimeError, "Could not verify Google credentials"):
            accounts.verified_credentials(cfg)

    def test_canceled_login_preserves_selection_and_both_cached_credentials(self):
        second_path = self.cache(self.second, marker="keep-second")
        previous = self.before(second_path)
        self.login_exit_code = 1
        with self.assertRaisesRegex(RuntimeError, "login did not finish"):
            backend.login(self.second, reauth=True)
        self.assert_unchanged(previous)
        self.assertEqual(self.assignment_calls, [self.first])

    def test_wrong_browser_account_preserves_selection_and_both_credentials(self):
        second_path = self.cache(self.second, marker="keep-second")
        previous = self.before(second_path)
        self.login_email = "wrong@example.com"
        with self.assertRaisesRegex(RuntimeError, "Google account mismatch"):
            backend.login(self.second, reauth=True)
        self.assert_unchanged(previous)
        self.assertEqual(self.assignment_calls, [self.first])

    def test_colab_permission_failure_preserves_selection_and_both_credentials(self):
        second_path = self.cache(self.second, marker="keep-second")
        previous = self.before(second_path)
        self.login_colab_denied = True
        with self.assertRaisesRegex(RuntimeError, "Colab permission denied"):
            backend.login(self.second, reauth=True)
        self.assert_unchanged(previous)

    def test_active_old_account_blocks_before_browser_login(self):
        previous = self.before()
        self.assignment_results[self.first] = [[SimpleNamespace(endpoint="old-running")]]
        with self.assertRaisesRegex(RuntimeError, "active Colab runtimes"):
            backend.login(self.second)
        self.assertEqual(self.login_calls, [])
        self.assert_unchanged(previous)

    def test_old_runtime_started_during_browser_login_blocks_commit(self):
        second_path = self.cache(self.second, marker="keep-second")
        previous = self.before(second_path)
        self.assignment_results[self.first] = [[], [SimpleNamespace(endpoint="newly-running")]]
        with self.assertRaisesRegex(RuntimeError, "active Colab runtimes"):
            backend.login(self.second, reauth=True)
        self.assertEqual(len(self.login_calls), 1)
        self.assertEqual(self.assignment_calls, [self.first, self.second, self.first])
        self.assert_unchanged(previous)

    def test_same_account_reauth_works_with_expired_old_credentials(self):
        self.cache(self.first, marker="expired-original", expired=True)
        result = backend.login(self.first, reauth=True)
        self.assertFalse(result["switched"])
        self.assertEqual(self.assignment_calls, [self.first])
        self.assertNotIn(self.first_path, self.loaded_paths)
        self.assertEqual(len(self.login_calls), 1)
        self.assertEqual(json.loads(self.first_path.read_text())["marker"], "new-browser-login")
        self.assertEqual(backend.config(), self.cfg)

    def test_status_reports_external_running_runtime_without_managed_mapping(self):
        assignments = [SimpleNamespace(endpoint="external-running", token="not-for-output")]
        with patch.object(session_state, "synced_state", return_value=({}, assignments)), \
             patch.object(backend, "remote_call") as remote_call:
            result = backend.status()
        self.assertFalse(result["active"])
        self.assertEqual(result["active_runtimes"], 1)
        self.assertEqual(result["next"], "colab-persist sessions")
        self.assertEqual(result["account"], self.first)
        remote_call.assert_not_called()
        self.assertNotIn("not-for-output", json.dumps(result))

    def test_sessions_distinguishes_managed_other_named_and_external_runtimes(self):
        sessions = {"training": SimpleNamespace(endpoint="managed"),
                    "other": SimpleNamespace(endpoint="other-named")}
        assignments = [SimpleNamespace(endpoint=endpoint, token="not-for-output")
                       for endpoint in ("managed", "other-named", "external")]
        with patch.object(session_state, "synced_state", return_value=(sessions, assignments)):
            result = backend.list_sessions()
        self.assertEqual(result["active_runtimes"], 3)
        self.assertEqual(result["sessions"], [
            {"endpoint": "managed", "name": "training", "managed": True},
            {"endpoint": "other-named", "name": "other", "managed": False},
            {"endpoint": "external", "name": None, "managed": False},
        ])
        self.assertNotIn("not-for-output", json.dumps(result))

    def configure_args(self, **overrides):
        return SimpleNamespace(**{"email": self.first, "gpu": None, "session": None,
                                  "key": None, "ssh_config": None, "snapshot_limit_gib": None,
                                  **overrides})

    def test_configure_refuses_email_switch_before_subprocess_or_config_write(self):
        previous = self.before()
        with self.assertRaisesRegex(ValueError, "configure does not switch Google credentials"):
            client.configure(self.configure_args(email=self.second))
        self.assertEqual(self.login_calls, [])
        self.assertEqual(self.assignment_calls, [])
        self.assert_unchanged(previous)

    def test_configure_refuses_session_or_key_changes_while_runtime_active(self):
        for change in ({"session": "replacement"}, {"key": str(self.root / "replacement-key")}):
            with self.subTest(change=change):
                previous = self.before()
                self.assignment_results[self.first] = [[SimpleNamespace(endpoint="running")]]
                with self.assertRaisesRegex(RuntimeError, "before changing the session name or SSH key"):
                    client.configure(self.configure_args(**change))
                self.assertEqual(self.login_calls, [])
                self.assert_unchanged(previous)


if __name__ == "__main__":
    unittest.main()
