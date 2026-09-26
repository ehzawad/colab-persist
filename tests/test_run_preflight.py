import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from colab_persist import backend, client, remote


class RunPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.script = self.root / "train.py"
        self.script.write_text("print('training')\n")
        self.settings = {"snapshot_limit_bytes": 1024}
        config = patch.object(backend, "config", return_value=self.settings)
        config.start()
        self.addCleanup(config.stop)

    def assert_cli_rejected_without_runtime(self, arguments):
        with patch.object(client.sys, "argv", ["colab-persist", "run", *arguments]), \
                patch.object(client, "invoke") as invoke, \
                patch.object(backend, "mount_drive") as mount, \
                patch.object(backend, "identity") as identity, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                client.main()
            self.assertEqual(stopped.exception.code, 1)
        invoke.assert_not_called()
        mount.assert_not_called()
        identity.assert_not_called()

    def test_missing_or_non_python_script_never_allocates(self):
        text = self.root / "notes.txt"
        text.write_text("not Python")
        directory = self.root / "directory.py"
        directory.mkdir()
        for script in (self.root / "missing.py", text, directory):
            with self.subTest(script=script):
                self.assert_cli_rejected_without_runtime([str(script)])

    def test_bad_project_or_interval_never_allocates(self):
        for options in (["--project", "../escape"], ["--project", "space here"],
                        ["--checkpoint-seconds", "0"], ["--checkpoint-seconds", "9"]):
            with self.subTest(options=options):
                self.assert_cli_rejected_without_runtime([str(self.script), *options])

    def test_oversized_script_and_source_never_allocate_or_hash(self):
        large = self.root / "large.py"
        large.write_bytes(b"#" * 1025)
        with patch.object(remote, "digest", side_effect=AssertionError("Preflight must not hash")):
            self.assert_cli_rejected_without_runtime([str(large)])
            self.assert_cli_rejected_without_runtime([str(self.script), "--source", str(self.root)])

    def test_explicit_source_must_be_a_project_containing_the_script(self):
        other = self.root / "other-project"
        other.mkdir()
        for source in (other, self.script, self.root / "missing", Path("/"), Path.home()):
            with self.subTest(source=source):
                self.assert_cli_rejected_without_runtime([str(self.script), "--source", str(source)])

    def test_excluded_entry_script_never_allocates(self):
        excluded = self.root / ".config"
        excluded.mkdir()
        script = excluded / "train.py"
        script.write_text("pass\n")
        self.assert_cli_rejected_without_runtime([str(script), "--source", str(self.root)])

    def test_metadata_check_skips_exclusions_links_and_special_files(self):
        cache = self.root / ".venv"
        cache.mkdir()
        (cache / "weights").write_bytes(b"x" * 2048)
        (self.root / ".env").write_bytes(b"x" * 2048)
        outside = self.root / "outside"
        outside.mkdir()
        large = outside / "large.bin"
        large.write_bytes(b"x" * 2048)
        source = self.root / "project"
        source.mkdir()
        script = source / "train.py"
        script.write_text("pass\n")
        (source / "linked-file").symlink_to(large)
        (source / "linked-dir").symlink_to(outside)
        (source / ".venv").symlink_to(cache)
        (source / ".env").write_bytes(b"x" * 2048)
        (source / "download.partial").write_bytes(b"x" * 2048)
        backend.os.mkfifo(source / "pipe")
        with patch.object(Path, "open", side_effect=AssertionError("Preflight must not read contents")), \
                patch.object(remote, "digest", side_effect=AssertionError("Preflight must not hash")):
            checked = backend.validate_run_inputs(script, "valid-project", source, 10)
        self.assertEqual(checked, (script, source, Path("train.py")))

    def test_valid_cli_request_proceeds_to_runtime_and_run(self):
        calls = []

        def invoke(name, **arguments):
            calls.append((name, arguments))
            if name == "start_runtime":
                return {"session": "test", "gpu": "L4"}
            return {"exit_code": 0}

        arguments = ["colab-persist", "run", str(self.script), "--source", str(self.root),
                     "--project", "new-project", "--checkpoint-seconds", "10", "--", "--steps", "2"]
        with patch.object(client.sys, "argv", arguments), \
                patch.object(client, "invoke", side_effect=invoke), \
                patch.object(backend, "mount_drive") as mount, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client.main()
        mount.assert_called_once_with()
        self.assertEqual([name for name, _ in calls], ["start_runtime", "run_script"])
        self.assertEqual(calls[1][1]["arguments"], ["--steps", "2"])
        self.assertEqual(calls[1][1]["checkpoint_seconds"], 10)
        self.assertEqual(calls[1][1]["project"], "new-project")

    def test_mcp_backend_rejects_invalid_input_before_lock_or_connection(self):
        cases = [dict(script_path=self.root / "missing.py"),
                 dict(script_path=self.script, project="../escape"),
                 dict(script_path=self.script, checkpoint_seconds=9)]
        for arguments in cases:
            with self.subTest(arguments=arguments), \
                    patch.object(backend, "operation_lock") as lock, \
                    patch.object(backend, "require_session") as connection:
                with self.assertRaises((ValueError, OSError)):
                    backend.run_script(**arguments)
                lock.assert_not_called()
                connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
