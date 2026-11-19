"""Private per-account credentials, scopes, quota projects, config migration and safe switching."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import accounts, backend, client, colab_api, drive_api, remote

import fakes

FIRST, SECOND = "first@example.com", "second@example.com"


class FakeAuthorizedSession:
    """Answers the userinfo request from the fake credential document."""

    def __init__(self, credentials):
        self.credentials = credentials

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get(self, url, timeout):
        assert url == accounts.USERINFO
        if self.credentials.get("expired"):
            raise RuntimeError("fake expired credentials")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"email": self.credentials["email"]})


class AccountModuleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_normalize_email(self):
        self.assertEqual(accounts.normalize_email("First@Example.COM"), FIRST)
        for bad in ("", "no-at", "a@b@c", "@example.com", "first@", "has space@example.com",
                    "-first@example.com", None, 5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                accounts.normalize_email(bad)

    def test_scopes_default_to_drive_file(self):
        self.assertEqual(accounts.SCOPES, accounts.login_scopes())
        self.assertIn(accounts.DRIVE_FILE_SCOPE, accounts.login_scopes())
        self.assertNotIn(accounts.DRIVE_FULL_SCOPE, accounts.login_scopes())
        self.assertIn(accounts.DRIVE_FULL_SCOPE, accounts.login_scopes(full_drive=True))
        self.assertNotIn(accounts.DRIVE_FILE_SCOPE, accounts.login_scopes(full_drive=True))
        for scopes in (accounts.login_scopes(), accounts.login_scopes(True)):
            self.assertIn(accounts.COLAB_SCOPE, scopes)
            self.assertIn("https://www.googleapis.com/auth/userinfo.email", scopes)

    def test_account_directory_is_private_and_named_without_the_email(self):
        directory = accounts.account_dir(self.root, "First@Example.com")
        self.assertEqual(directory.name, hashlib.sha256(FIRST.encode()).hexdigest())
        self.assertEqual(directory, accounts.account_dir(self.root, FIRST))
        self.assertNotIn("example", str(directory))
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(directory.parent.stat().st_mode), 0o700)
        self.assertNotEqual(directory, accounts.account_dir(self.root, SECOND))

    def test_metadata_merges_and_is_private(self):
        self.assertEqual(accounts.meta(self.root, FIRST), {})
        accounts.save_meta(self.root, FIRST, quota_project="demo-project")
        merged = accounts.save_meta(self.root, FIRST, full_drive=True)
        self.assertEqual(merged, {"quota_project": "demo-project", "full_drive": True})
        self.assertEqual(accounts.meta(self.root, FIRST), merged)
        self.assertEqual(accounts.meta(self.root, SECOND), {})
        path = accounts.account_dir(self.root, FIRST) / "meta.json"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_credentials_path(self):
        self.assertIsNone(accounts.credentials_path({}))
        self.assertTrue(accounts.credentials_path({"credentials_file": "relative.json"}).is_absolute())

    def test_verified_credentials_checks_the_google_identity(self):
        path = self.root / "credentials.json"
        path.write_text(json.dumps({"email": FIRST}))
        loads = []

        def load(filename, scopes):
            loads.append((filename, scopes))
            return json.loads(Path(filename).read_text()), None
        cfg = {"expected_email": FIRST, "credentials_file": str(path)}
        with patch.object(accounts.google.auth, "load_credentials_from_file", side_effect=load), \
             patch.object(accounts, "AuthorizedSession", FakeAuthorizedSession):
            self.assertEqual(accounts.verified_credentials(cfg), {"email": FIRST})
            self.assertEqual(loads[-1], (str(path), list(accounts.SCOPES)))
            full = list(accounts.login_scopes(True))
            accounts.verified_credentials({**cfg, "scopes": full})
            self.assertEqual(loads[-1][1], full)
            with self.assertRaisesRegex(RuntimeError, f"Google account mismatch: expected {SECOND}, got {FIRST}"):
                accounts.verified_credentials({**cfg, "expected_email": SECOND})
            path.write_text(json.dumps({"email": FIRST, "expired": True}))
            with self.assertRaisesRegex(RuntimeError, "Could not verify Google credentials.*--reauth"):
                accounts.verified_credentials(cfg)

    def test_missing_private_credentials_do_not_fall_back_to_the_shell(self):
        cfg = {"expected_email": FIRST, "credentials_file": str(self.root / "missing.json")}
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": str(self.root / "unrelated.json")}), \
             patch.object(accounts.google.auth, "load_credentials_from_file") as load:
            with self.assertRaisesRegex(RuntimeError, "no private colab-persist login yet.*--reauth"):
                accounts.verified_credentials(cfg)
            with self.assertRaisesRegex(RuntimeError, "Saved account credentials are missing"):
                accounts.drive_token(cfg, "demo-project", "Colab-Persist")
            load.assert_not_called()

    def test_drive_token_is_narrowed_short_lived_and_has_no_refresh_token(self):
        path = self.root / "credentials.json"
        path.write_text(json.dumps({"refresh_token": "fake-refresh-token", "client_id": "fake-client",
                                    "client_secret": "fake-secret", "type": "authorized_user"}))
        issued = []

        class FakeCredentials:
            token, expiry = "fake-narrowed-access-token", None

            @classmethod
            def from_authorized_user_info(cls, info, scopes):
                issued.append(scopes)
                credentials = cls()
                credentials.expiry = __import__("datetime").datetime(2030, 1, 1, 12, 0, 0)
                return credentials

            def refresh(self, request):
                self.refreshed = True
        cfg = {"expected_email": FIRST, "credentials_file": str(path)}
        with patch.object(accounts, "Credentials", FakeCredentials), patch.object(accounts, "Request"):
            token = accounts.drive_token(cfg, "demo-project", "Colab-Persist")
            accounts.drive_token(cfg, "demo-project", "Colab-Persist", full_drive=True)
        self.assertEqual(issued, [[accounts.DRIVE_FILE_SCOPE], [accounts.DRIVE_FULL_SCOPE]])
        self.assertEqual(token, {"token": "fake-narrowed-access-token", "quota_project": "demo-project",
                                 "expires_at": 1893499200, "folder": "Colab-Persist"})
        self.assertNotIn("fake-refresh-token", json.dumps(token))

    def test_drive_token_refusal_asks_for_a_new_login(self):
        path = self.root / "credentials.json"
        path.write_text(json.dumps({"refresh_token": "x", "client_id": "y", "client_secret": "z"}))

        class Refusing:
            @classmethod
            def from_authorized_user_info(cls, info, scopes):
                return cls()

            def refresh(self, request):
                raise ValueError("invalid_scope")
        with patch.object(accounts, "Credentials", Refusing), patch.object(accounts, "Request"):
            with self.assertRaisesRegex(RuntimeError, "predates colab-persist 0.4.*--email first@example.com --reauth"):
                accounts.drive_token({"expected_email": FIRST, "credentials_file": str(path)}, "p", "f")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        directory = patch.object(backend, "CONFIG_DIR", self.root)
        directory.start()
        self.addCleanup(directory.stop)

    def write(self, value):
        (self.root / "config.json").write_text(json.dumps(value))

    def test_v03_configuration_migrates_to_the_api_folder(self):
        for stored, legacy in (({"drive_folder": "Colab-CUDA"}, "Colab-CUDA"), ({"drive_folder": "Training"}, "Training"),
                               ({}, "Colab-CUDA")):
            with self.subTest(stored=stored):
                self.write({"expected_email": FIRST, **stored})
                cfg = backend.config()
                self.assertEqual(cfg["drive_folder"], "Colab-Persist")
                self.assertEqual(cfg["legacy_drive_folder"], legacy)
                self.assertEqual(cfg["storage"], 2)
                self.assertEqual(cfg["scopes"], list(accounts.SCOPES))

    def test_v04_configuration_is_kept(self):
        self.write({"expected_email": FIRST, "storage": 2, "drive_folder": "My-Folder", "legacy_drive_folder": "Old"})
        cfg = backend.config()
        self.assertEqual((cfg["drive_folder"], cfg["legacy_drive_folder"]), ("My-Folder", "Old"))

    def test_scopes_follow_the_accounts_drive_choice(self):
        self.write({"expected_email": FIRST, "storage": 2})
        accounts.save_meta(self.root, FIRST, full_drive=True)
        self.assertIn(accounts.DRIVE_FULL_SCOPE, backend.config()["scopes"])

    def test_invalid_configurations(self):
        with self.assertRaisesRegex(RuntimeError, "Configure your account first"):
            backend.config()
        cases = [({"gpu": "L4"}, RuntimeError, "expected_email"),
                 ({"expected_email": "not-an-email"}, ValueError, "email"),
                 ({"expected_email": FIRST, "storage": 2, "drive_folder": "../escape"}, ValueError, "Project names"),
                 ({"expected_email": FIRST, "session": "has space"}, ValueError, "Project names"),
                 ({"expected_email": FIRST, "snapshot_limit_bytes": 0}, ValueError, "snapshot_limit_bytes"),
                 ({"expected_email": FIRST, "snapshot_limit_bytes": True}, ValueError, "snapshot_limit_bytes")]
        for value, error, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(error, message):
                self.write(value)
                backend.config()

    def test_quota_project_is_required_for_api_access(self):
        self.write({"expected_email": FIRST, "storage": 2})
        with self.assertRaisesRegex(RuntimeError, "setup --project"):
            backend.quota_project(backend.config())
        accounts.save_meta(self.root, FIRST, quota_project="demo-project")
        self.assertEqual(backend.quota_project(backend.config()), "demo-project")


class LoginTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "persist"
        self.config_dir.mkdir()
        self.first_path = self.cache(FIRST, marker="original-first")
        accounts.save_meta(self.config_dir, FIRST, quota_project="first-project")
        self.cfg = {"expected_email": FIRST, "credentials_file": str(self.first_path), "session": "training",
                    "gpu": "A100", "drive_folder": "Colab-Persist", "legacy_drive_folder": "Colab-CUDA",
                    "storage": 2, "ssh_identity": str(self.root / "fake-key")}
        self.config_path = self.config_dir / "config.json"
        self.config_path.write_text(json.dumps(self.cfg))
        self.loaded = []
        self.listed = []          # Account emails whose runtimes were listed.
        self.runtime_results = {}  # email -> list of results for successive list_runtimes calls
        self.drive_emails = {}     # email -> email Drive reports (to simulate a mismatch)
        self.minted = []
        self.login_calls = []
        self.login_exit_code = 0
        self.login_email = None
        self.login_extra = {}
        patches = [patch.object(backend, "CONFIG_DIR", self.config_dir),
                   patch.object(accounts.google.auth, "load_credentials_from_file", side_effect=self.load),
                   patch.object(accounts, "AuthorizedSession", FakeAuthorizedSession),
                   patch.object(colab_api, "ColabAPI", side_effect=self.make_api),
                   patch.object(accounts, "drive_token", side_effect=self.mint),
                   patch.object(drive_api, "DriveClient", side_effect=self.make_drive),
                   patch.object(backend.subprocess, "run", side_effect=self.gcloud)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        backend._TOKENS.clear()
        self.addCleanup(backend._TOKENS.clear)

    # -- fakes ---------------------------------------------------------------------------
    def cache(self, email, **data):
        path = accounts.account_dir(self.config_dir, email) / "credentials.json"
        path.write_text(json.dumps({"email": email, **data}))
        path.chmod(0o600)
        return path

    def load(self, filename, scopes):
        self.loaded.append(Path(filename))
        return json.loads(Path(filename).read_text()), None

    def make_api(self, credentials, project, **_ignored):
        email = credentials["email"]

        def list_runtimes():
            self.listed.append(email)
            if credentials.get("colab_denied"):
                raise RuntimeError("fake Colab permission denied")
            results = self.runtime_results.get(email)
            return results.pop(0) if results else []
        return SimpleNamespace(list_runtimes=list_runtimes, project=project)

    def mint(self, cfg, project, folder, *, full_drive=False):
        email = json.loads(Path(cfg["credentials_file"]).read_text())["email"]
        self.minted.append((email, project, folder, full_drive))
        return {"token": "token-for:" + email, "quota_project": project, "expires_at": time.time() + 3600,
                "folder": folder}

    def make_drive(self, tokens, **_ignored):
        email = tokens.get()[0].split(":", 1)[1]
        return fakes.FakeDrive(email=self.drive_emails.get(email, email))

    def gcloud(self, command, *, env):
        self.assertEqual(command[:4], ["gcloud", "auth", "application-default", "login"])
        self.login_calls.append((command, env.copy()))
        staged = Path(env["CLOUDSDK_CONFIG"]) / "application_default_credentials.json"
        staged.write_text(json.dumps({"email": self.login_email or command[4], "marker": "new-browser-login",
                                      **self.login_extra}))
        return SimpleNamespace(returncode=self.login_exit_code)

    def before(self, *extra):
        return {path: path.read_bytes() for path in (self.config_path, self.first_path, *extra)}

    def assert_unchanged(self, previous):
        for path, content in previous.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def selected(self):
        return {**self.cfg, "scopes": list(accounts.SCOPES)}

    # -- switching -------------------------------------------------------------------------
    def test_switch_first_second_first_uses_cached_private_credentials(self):
        second_path = self.cache(SECOND, marker="original-second")
        accounts.save_meta(self.config_dir, SECOND, quota_project="second-project")
        first_bytes, second_bytes = self.first_path.read_bytes(), second_path.read_bytes()
        result = backend.login(SECOND)
        self.assertTrue(result["switched"])
        self.assertTrue(result["drive_ready"])
        self.assertEqual(result["drive_scope"], "drive.file")
        self.assertEqual(backend.config()["expected_email"], SECOND)
        self.assertEqual(backend.config()["credentials_file"], str(second_path))
        self.assertEqual(self.listed, [FIRST, SECOND, FIRST])
        result = backend.login(FIRST)
        self.assertTrue(result["switched"])
        self.assertEqual(backend.config(), self.selected())
        self.assertEqual(self.login_calls, [])
        self.assertEqual((self.first_path.read_bytes(), second_path.read_bytes()), (first_bytes, second_bytes))
        self.assertIn(self.first_path, self.loaded)
        self.assertIn(second_path, self.loaded)
        self.assertEqual(stat.S_IMODE(self.config_path.stat().st_mode), 0o600)
        self.assertEqual([m[:2] for m in self.minted], [(SECOND, "second-project"), (FIRST, "first-project")])

    def test_browser_login_is_isolated_from_the_shell_and_global_gcloud(self):
        original_adc = self.root / "unrelated-adc.json"
        original_sdk = self.root / "unrelated-gcloud"
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": str(original_adc),
                                     "CLOUDSDK_CONFIG": str(original_sdk)}):
            result = backend.login(SECOND, no_launch_browser=True, project="second-project")
            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], str(original_adc))
            self.assertEqual(os.environ["CLOUDSDK_CONFIG"], str(original_sdk))
        self.assertFalse(result["global_gcloud_credentials_changed"])
        command, environment = self.login_calls[0]
        staging = Path(environment["CLOUDSDK_CONFIG"])
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", environment)
        self.assertEqual(staging.parent, accounts.account_dir(self.config_dir, SECOND))
        self.assertFalse(staging.exists())
        self.assertFalse(original_adc.exists())
        self.assertFalse(original_sdk.exists())
        self.assertIn("--no-launch-browser", command)
        self.assertIn("--disable-quota-project", command)
        self.assertIn("--scopes=" + ",".join(accounts.SCOPES), command)
        self.assertFalse(any(arg.startswith("--client-id-file") for arg in command))
        selected = Path(backend.config()["credentials_file"])
        self.assertEqual(selected, accounts.account_dir(self.config_dir, SECOND) / "credentials.json")
        self.assertEqual(json.loads(selected.read_text())["email"], SECOND)
        self.assertEqual(stat.S_IMODE(selected.stat().st_mode), 0o600)
        self.assertEqual(accounts.meta(self.config_dir, SECOND), {"full_drive": False, "quota_project": "second-project"})

    def test_full_drive_login_uses_the_users_oauth_client(self):
        client_file = self.root / "client_secret.json"
        client_file.write_text("{}")
        result = backend.login(FIRST, client_id_file=str(client_file))
        command = self.login_calls[0][0]
        self.assertIn("--client-id-file=" + str(client_file.absolute()), command)
        self.assertIn("--scopes=" + ",".join(accounts.login_scopes(True)), command)
        self.assertEqual(result["drive_scope"], "drive")
        self.assertTrue(accounts.meta(self.config_dir, FIRST)["full_drive"])
        self.assertTrue(self.minted[-1][3])
        self.assertIn(accounts.DRIVE_FULL_SCOPE, backend.config()["scopes"])
        # Returning to drive.file needs a fresh consent with gcloud's own client.
        backend.login(FIRST)
        self.assertEqual(len(self.login_calls), 2)
        self.assertFalse(accounts.meta(self.config_dir, FIRST)["full_drive"])
        self.assertEqual(backend.config()["scopes"], list(accounts.SCOPES))

    def test_project_is_remembered_and_its_absence_is_explained(self):
        result = backend.login(SECOND, no_switch=True)
        self.assertFalse(result["drive_ready"])
        self.assertEqual(result["next"], "colab-persist setup --project YOUR_PROJECT_ID")
        self.assertNotIn(SECOND, [m[0] for m in self.minted])
        result = backend.login(SECOND, no_switch=True, project="second-project")
        self.assertTrue(result["drive_ready"])
        self.assertEqual(accounts.meta(self.config_dir, SECOND)["quota_project"], "second-project")

    def test_canceled_login_preserves_selection_and_both_cached_credentials(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        self.login_exit_code = 1
        with self.assertRaisesRegex(RuntimeError, "login did not finish"):
            backend.login(SECOND, reauth=True)
        self.assert_unchanged(previous)
        self.assertEqual(self.listed, [FIRST])

    def test_wrong_browser_account_preserves_selection_and_both_credentials(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        self.login_email = "wrong@example.com"
        with self.assertRaisesRegex(RuntimeError, "Google account mismatch"):
            backend.login(SECOND, reauth=True)
        self.assert_unchanged(previous)
        self.assertEqual(self.listed, [FIRST])

    def test_colab_permission_failure_preserves_selection_and_both_credentials(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        self.login_extra = {"colab_denied": True}
        with self.assertRaisesRegex(RuntimeError, "Colab permission denied"):
            backend.login(SECOND, reauth=True, project="second-project")
        self.assert_unchanged(previous)

    def test_drive_account_mismatch_preserves_selection(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        self.drive_emails[SECOND] = "someone-else@example.com"
        with self.assertRaisesRegex(RuntimeError, "Drive account mismatch.*Nothing was written"):
            backend.login(SECOND, reauth=True, project="second-project")
        self.assert_unchanged(previous)
        self.assertEqual(accounts.meta(self.config_dir, SECOND), {})

    def test_login_without_a_drive_grant_preserves_selection(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        with patch.object(accounts, "drive_token", side_effect=RuntimeError("Google did not issue a Drive token")):
            with self.assertRaisesRegex(RuntimeError, "did not issue a Drive token"):
                backend.login(SECOND, reauth=True, project="second-project")
        self.assert_unchanged(previous)

    def test_active_old_account_blocks_before_browser_login(self):
        previous = self.before()
        self.runtime_results[FIRST] = [[fakes.make_runtime("cp-old-runtime")]]
        with self.assertRaisesRegex(RuntimeError, "active Colab runtimes"):
            backend.login(SECOND)
        self.assertEqual(self.login_calls, [])
        self.assert_unchanged(previous)

    def test_old_account_that_cannot_be_checked_blocks_switching(self):
        previous = self.before()
        accounts.save_meta(self.config_dir, FIRST, quota_project=None)
        with self.assertRaisesRegex(RuntimeError, "Could not verify that the current account has no running"):
            backend.login(SECOND)
        self.assertEqual(self.login_calls, [])
        self.assert_unchanged(previous)

    def test_old_runtime_started_during_browser_login_blocks_commit(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        self.runtime_results[FIRST] = [[], [fakes.make_runtime("cp-newly-running")]]
        with self.assertRaisesRegex(RuntimeError, "active Colab runtimes"):
            backend.login(SECOND, reauth=True, project="second-project")
        self.assertEqual(len(self.login_calls), 1)
        self.assertEqual(self.listed, [FIRST, SECOND, FIRST])
        self.assert_unchanged(previous)

    def test_same_account_reauth_works_with_expired_old_credentials(self):
        self.cache(FIRST, marker="expired-original", expired=True)
        result = backend.login(FIRST, reauth=True)
        self.assertFalse(result["switched"])
        self.assertEqual(self.listed, [FIRST])
        self.assertNotIn(self.first_path, self.loaded)
        self.assertEqual(len(self.login_calls), 1)
        self.assertEqual(json.loads(self.first_path.read_text())["marker"], "new-browser-login")
        self.assertEqual(backend.config(), self.selected())

    def test_login_clears_cached_drive_tokens(self):
        backend._TOKENS[(FIRST, "Colab-Persist")] = {"token": "stale", "expires_at": time.time() + 9999}
        backend.login(FIRST)
        self.assertEqual(backend._TOKENS, {})

    # -- save without switching -----------------------------------------------------------
    def test_save_login_without_switch_preserves_selection_and_can_be_selected_later(self):
        previous = self.before()
        result = backend.login(SECOND, no_switch=True, project="second-project")
        self.assert_unchanged(previous)
        self.assertTrue(result["saved_only"])
        self.assertFalse(result["switched"])
        self.assertTrue(result["account_verified"])
        self.assertEqual(result["account"], SECOND)
        self.assertEqual(result["selected_account"], FIRST)
        self.assertEqual(self.listed, [SECOND])
        saved = accounts.account_dir(self.config_dir, SECOND) / "credentials.json"
        self.assertEqual(json.loads(saved.read_text())["email"], SECOND)
        self.assertEqual(stat.S_IMODE(saved.stat().st_mode), 0o600)
        self.assertEqual(len(self.login_calls), 1)
        selected = backend.login(SECOND)
        self.assertTrue(selected["switched"])
        self.assertFalse(selected["saved_only"])
        self.assertEqual(selected["selected_account"], SECOND)
        self.assertEqual(backend.config()["expected_email"], SECOND)
        self.assertEqual(backend.config()["credentials_file"], str(saved))
        self.assertEqual(len(self.login_calls), 1)
        self.assertEqual(self.first_path.read_bytes(), previous[self.first_path])

    def test_save_login_does_not_authenticate_or_query_the_old_account(self):
        self.cache(FIRST, marker="old-expired", expired=True)
        self.runtime_results[FIRST] = [[fakes.make_runtime("cp-old-runtime")]]
        previous = self.before()
        backend.login(SECOND, no_switch=True, project="second-project")
        self.assert_unchanged(previous)
        self.assertNotIn(self.first_path, self.loaded)
        self.assertEqual(self.listed, [SECOND])

    def test_cached_save_login_does_not_change_configuration_or_reauthenticate(self):
        second_path = self.cache(SECOND, marker="cached-second")
        previous = self.before(second_path)
        result = backend.login(SECOND, no_switch=True, project="second-project")
        self.assertTrue(result["saved_only"])
        self.assert_unchanged(previous)
        self.assertEqual(self.login_calls, [])
        self.assertEqual(self.listed, [SECOND])

    def test_wrong_account_during_save_only_preserves_existing_logins(self):
        second_path = self.cache(SECOND, marker="keep-second")
        previous = self.before(second_path)
        self.login_email = "wrong@example.com"
        with self.assertRaisesRegex(RuntimeError, "Google account mismatch"):
            backend.login(SECOND, reauth=True, no_switch=True)
        self.assert_unchanged(previous)
        self.assertEqual(self.listed, [])

    def test_cli_save_only_flag_reaches_login_without_switching(self):
        with patch.object(sys, "argv", ["colab-persist", "login", "--email", SECOND, "--no-switch",
                                        "--no-launch-browser", "--project", "second-project"]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            client.main()
        result = json.loads(output.getvalue())
        self.assertTrue(result["saved_only"])
        self.assertFalse(result["switched"])
        self.assertEqual(backend.config()["expected_email"], FIRST)
        self.assertEqual(self.listed, [SECOND])
        self.assertIn("--no-launch-browser", self.login_calls[0][0])

    # -- configure ------------------------------------------------------------------------
    def configure_args(self, **overrides):
        return SimpleNamespace(**{"email": FIRST, "gpu": None, "session": None, "key": None, "ssh_config": None,
                                  "snapshot_limit_gib": None, **overrides})

    def test_configure_refuses_email_switch_before_subprocess_or_config_write(self):
        previous = self.before()
        with self.assertRaisesRegex(ValueError, "configure does not switch Google credentials"):
            client.configure(self.configure_args(email=SECOND))
        self.assertEqual(self.login_calls, [])
        self.assertEqual(self.listed, [])
        self.assert_unchanged(previous)

    def test_configure_refuses_session_or_key_changes_while_runtime_active(self):
        for change in ({"session": "replacement"}, {"key": str(self.root / "replacement-key")}):
            with self.subTest(change=change):
                previous = self.before()
                with patch.object(backend, "runtime_info", return_value=fakes.make_runtime("cp-training-abc123")):
                    with self.assertRaisesRegex(RuntimeError, "before changing the session name or SSH key"):
                        client.configure(self.configure_args(**change))
                self.assertEqual(self.login_calls, [])
                self.assert_unchanged(previous)


class ConfigureTests(unittest.TestCase):
    def test_changing_snapshot_cap_preserves_settings_and_completes_the_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "custom-key"
            key.write_text("test key fixture")
            original = {"expected_email": "test@example.com", "gpu": "A100", "session": "training",
                        "ssh_identity": str(key), "drive_folder": "Training", "ssh_alias_config": "custom.conf"}
            remote.write_json(root / "config.json", original)
            args = SimpleNamespace(email="test@example.com", gpu=None, session=None, key=None,
                                   ssh_config=None, snapshot_limit_gib=8)
            with patch.object(backend, "CONFIG_DIR", root), patch.object(client.subprocess, "run") as command:
                client.configure(args)
                command.assert_not_called()
            self.assertEqual(json.loads((root / "config.json").read_text()),
                             {**original, "snapshot_limit_bytes": 8 * remote.GIB, "storage": 2,
                              "drive_folder": "Colab-Persist", "legacy_drive_folder": "Training"})

    def test_first_configuration_creates_a_key_and_uses_the_api_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "keys" / "id_colab"

            def keygen(command, **_kwargs):
                Path(command[-1]).write_text("generated key fixture")
                return SimpleNamespace(returncode=0)
            args = SimpleNamespace(email="Test@Example.com", gpu="T4", session=None, key=str(key),
                                   ssh_config=str(root / "colab.conf"), snapshot_limit_gib=None)
            with patch.object(backend, "CONFIG_DIR", root / "config"), \
                 patch.object(client.subprocess, "run", side_effect=keygen) as command:
                result = client.configure(args)
            self.assertEqual(command.call_args.args[0][:3], ["ssh-keygen", "-t", "ed25519"])
            stored = json.loads((root / "config" / "config.json").read_text())
            self.assertEqual(stored["expected_email"], "test@example.com")
            self.assertEqual((stored["drive_folder"], stored["legacy_drive_folder"], stored["storage"]),
                             ("Colab-Persist", "Colab-CUDA", 2))
            self.assertEqual((stored["gpu"], stored["session"]), ("T4", "cuda"))
            self.assertEqual(stat.S_IMODE(key.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((root / "config" / "config.json").stat().st_mode), 0o600)
            self.assertEqual(result["drive_folder"], "My Drive/Colab-Persist")


class SetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "config.json").write_text(json.dumps({"expected_email": FIRST, "storage": 2}))
        self.api = fakes.FakeColabAPI(tier="SUBSCRIPTION_TIER_PRO")
        self.drive = fakes.FakeDrive(FIRST)
        for item in (patch.object(backend, "CONFIG_DIR", self.root),
                     patch.object(accounts, "verified_credentials", return_value=object()),
                     patch.object(colab_api, "ColabAPI", return_value=self.api),
                     patch.object(backend, "local_drive", side_effect=lambda cfg: self.drive)):
            item.start()
            self.addCleanup(item.stop)

    def test_setup_enables_free_apis_and_remembers_the_project(self):
        backend._TOKENS[(FIRST, "Colab-Persist")] = {"token": "stale", "expires_at": time.time() + 9999}
        result = backend.setup("  demo-project ")
        self.assertEqual(self.api.enabled, ["demo-project"])
        self.assertEqual(accounts.meta(self.root, FIRST)["quota_project"], "demo-project")
        self.assertEqual(result["colab_tier"], "pro")
        self.assertTrue(result["drive_ready"])
        self.assertEqual(result["services"]["drive.googleapis.com"], "ENABLED")
        self.assertEqual(backend._TOKENS, {})
        self.assertEqual(json.loads((self.root / "config.json").read_text())["drive_folder"], "Colab-Persist")

    def test_setup_reports_a_drive_mismatch_without_failing(self):
        self.drive.email = SECOND
        result = backend.setup("demo-project")
        self.assertFalse(result["drive_ready"])
        self.assertIn("Drive account mismatch", result["drive_error"])
        self.assertTrue(result["project_saved"])

    def test_setup_requires_a_project(self):
        for bad in ("", "   ", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "project ID"):
                backend.setup(bad)
        self.assertEqual(self.api.enabled, [])


if __name__ == "__main__":
    unittest.main()
