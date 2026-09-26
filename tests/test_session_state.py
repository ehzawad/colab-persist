from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_cli.client import RuntimeProxyInfo
from colab_cli.state import SessionState, StateStore

from colab_persist import session_state


class SessionIsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "persist"
        self.home = self.root / "home"
        self.home.mkdir()
        self.legacy_path = self.home / ".config" / "colab-cli" / "sessions.json"
        self.home_patch = patch.object(session_state.Path, "home", return_value=self.home)
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        self.credentials = object()
        self.cfg = {"expected_email": "first@example.com", "session": "cuda"}

    @staticmethod
    def item(endpoint, name="cuda"):
        return SessionState(name=name, endpoint=endpoint, token="old-test-token",
                            url="https://old.example.test", accelerator="L4")

    @staticmethod
    def assignment(endpoint):
        return SimpleNamespace(
            endpoint=endpoint,
            runtime_proxy_info=RuntimeProxyInfo(
                token="refreshed-test-token", url="https://runtime.example.test",
                tokenExpiresInSeconds=3600))

    def sync(self, assignments, cfg=None, error=None):
        with patch.object(session_state, "AuthorizedSession") as authorized, \
             patch.object(session_state, "Client") as client:
            client.return_value.list_assignments.return_value = assignments
            client.return_value.list_assignments.side_effect = error
            try:
                return session_state.synced_state(self.config_dir, cfg or self.cfg,
                                                 self.credentials)
            finally:
                authorized.assert_called_once_with(self.credentials)
                self.assertIs(client.call_args.args[1], authorized.return_value)
                authorized.return_value.close.assert_called_once_with()

    def write_legacy(self, *items):
        self.legacy_path.parent.mkdir(parents=True, exist_ok=True)
        store = StateStore(str(self.legacy_path))
        for item in items:
            store.add(item)
        return self.legacy_path.read_bytes()

    def test_account_paths_are_isolated_normalized_and_private(self):
        first = session_state.session_path(self.config_dir, " First@Example.com ")
        same = session_state.session_path(self.config_dir, "first@example.com")
        second = session_state.session_path(self.config_dir, "second@example.com")
        self.assertEqual(first, same)
        self.assertNotEqual(first, second)
        self.assertNotIn("first@", str(first))
        self.assertEqual(stat.S_IMODE(first.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(first.parent.parent.stat().st_mode), 0o700)
        first.write_text("{}")
        first.chmod(0o644)
        session_state.session_path(self.config_dir, "first@example.com")
        self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)

    def test_verified_legacy_adoption_refreshes_only_private_copy(self):
        before = self.write_legacy(self.item("mine"), self.item("foreign", "other"))
        assignments = [self.assignment("mine")]
        sessions, result = self.sync(assignments)
        self.assertIs(result, assignments)
        self.assertEqual(list(sessions), ["cuda"])
        self.assertEqual(sessions["cuda"].token, "refreshed-test-token")
        self.assertIsNotNone(sessions["cuda"].token_expires_at)
        self.assertEqual(self.legacy_path.read_bytes(), before)
        path = session_state.session_path(self.config_dir, self.cfg["expected_email"])
        self.assertEqual(StateStore(str(path)).get("cuda").endpoint, "mine")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_foreign_legacy_mapping_is_not_adopted_or_pruned(self):
        before = self.write_legacy(self.item("other-account"))
        sessions, _ = self.sync([self.assignment("my-unmanaged-runtime")])
        self.assertEqual(sessions, {})
        self.assertEqual(self.legacy_path.read_bytes(), before)
        path = session_state.session_path(self.config_dir, self.cfg["expected_email"])
        self.assertFalse(path.exists())

    def test_existing_verified_private_mapping_wins_over_legacy(self):
        before = self.write_legacy(self.item("legacy"))
        path = session_state.session_path(self.config_dir, self.cfg["expected_email"])
        StateStore(str(path)).add(self.item("private"))
        sessions, _ = self.sync([self.assignment("legacy"), self.assignment("private")])
        self.assertEqual(sessions["cuda"].endpoint, "private")
        self.assertEqual(self.legacy_path.read_bytes(), before)

    def test_only_current_account_is_pruned_and_other_account_is_preserved(self):
        first_path = session_state.session_path(self.config_dir, self.cfg["expected_email"])
        second_path = session_state.session_path(self.config_dir, "second@example.com")
        StateStore(str(first_path)).add(self.item("expired"))
        StateStore(str(second_path)).add(self.item("second-live"))
        second_before = second_path.read_bytes()
        sessions, _ = self.sync([])
        self.assertEqual(sessions, {})
        self.assertEqual(StateStore(str(first_path)).list(), {})
        self.assertEqual(second_path.read_bytes(), second_before)

    def test_missing_legacy_store_is_not_created(self):
        self.sync([])
        self.assertFalse(self.legacy_path.parent.exists())

    def test_api_failures_preserve_state_and_do_not_adopt_legacy(self):
        legacy_before = self.write_legacy(self.item("legacy"))
        path = session_state.session_path(self.config_dir, self.cfg["expected_email"])
        StateStore(str(path)).add(self.item("private"))
        private_before = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "denied"):
            self.sync([], error=RuntimeError("denied"))
        self.assertEqual(path.read_bytes(), private_before)
        self.assertEqual(self.legacy_path.read_bytes(), legacy_before)

    def test_system_exit_from_auth_is_not_hidden_as_an_empty_account(self):
        with self.assertRaises(SystemExit):
            self.sync([], error=SystemExit(1))
        path = session_state.session_path(self.config_dir, self.cfg["expected_email"])
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
