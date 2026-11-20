"""Drive-backed checkpoint safety: the v0.3 guarantees on the v0.4 Drive API engine."""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import backend, drive_api, remote

import fakes

PROJECT = ["Colab-Persist", "projects", "demo"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flipped(data: bytes) -> bytes:
    """Same length, different content."""
    changed = bytearray(data)
    changed[len(changed) // 2] ^= 0xFF
    return bytes(changed)


def tar_bytes(members):
    """members: list of (TarInfo, bytes or None)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as target:
        for info, data in members:
            target.addfile(info, io.BytesIO(data) if data is not None else None)
    return buffer.getvalue()


def file_member(name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    return info, data


def v03_checkpoint(destination: Path, workspace: Path, project: str, identifier: str):
    """Write a checkpoint pair exactly as colab-persist 0.3 did on the Drive mount."""
    files = {}
    for path in sorted(p for p in workspace.rglob("*") if p.is_file()):
        info = path.stat()
        files[path.relative_to(workspace).as_posix()] = {
            "size": info.st_size, "sha256": sha256(path.read_bytes()), "mode": stat.S_IMODE(info.st_mode)}
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (identifier + ".tar.gz")
    with tarfile.open(archive, "w:gz", compresslevel=1) as target:
        for name in files:
            target.add(workspace / name, arcname=name, recursive=False)
    manifest = {"schema": 1, "project": project, "snapshot": identifier,
                "created_at": "2025-06-01T12:00:00.000000+00:00", "reason": "safe_stop", "files": files,
                "archive_sha256": sha256(archive.read_bytes()), "archive_bytes": archive.stat().st_size}
    path = destination / (identifier + ".json")
    with path.open("w") as target:  # v0.3's write_json serialization
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
    return manifest, archive, path


class StoreCase(unittest.TestCase):
    def setUp(self):
        fakes.fixed_umask(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.drive = fakes.FakeDrive()
        self.store = remote.Store(self.base / "vm1", drive=self.drive)
        self.store.prepare("demo")
        self.workspace = self.store.workspace("demo")

    def replacement(self, name="vm2", **kwargs):
        return remote.Store(self.base / name, drive=self.drive, **kwargs)

    def manifests(self):
        return sorted(n for n in self.drive.listing(PROJECT) if n.endswith(".json"))

    def archives(self):
        return sorted(n for n in self.drive.listing(PROJECT) if n.endswith(".tar.gz"))


class CheckpointTests(StoreCase):
    def test_snapshot_survives_replacement_workspace(self):
        (self.workspace / "source.py").write_text("print('hello GPU')\n")
        (self.workspace / "outputs" / "weights.bin").write_bytes(bytes(range(256)) * 40)
        manifest = self.store.snapshot("demo")
        self.assertEqual(manifest["durability"], remote.DURABLE)
        self.assertFalse(manifest["reused"])
        self.assertEqual(self.archives(), [manifest["snapshot"] + ".tar.gz"])
        self.assertEqual(self.manifests(), [manifest["snapshot"] + ".json"])
        replacement = self.replacement()
        result = replacement.prepare("demo")
        self.assertEqual(result["restored"]["snapshot"], manifest["snapshot"])
        self.assertEqual(remote.inventory(replacement.workspace("demo")), remote.inventory(self.workspace))

    def test_unchanged_snapshot_is_reused_without_uploading(self):
        (self.workspace / "data").write_text("stable")
        first = self.store.snapshot("demo")
        second = self.store.snapshot("demo")
        self.assertEqual(first["snapshot"], second["snapshot"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["durability"], remote.DURABLE)
        self.assertEqual(len(self.drive.upload_names()), 2)

    def test_reuse_requires_the_archive_to_still_match_in_drive(self):
        (self.workspace / "data").write_text("stable")
        first = self.store.snapshot("demo")
        self.drive.trash(first["archive_file_id"])
        second = self.store.snapshot("demo")
        self.assertFalse(second["reused"])
        self.assertNotEqual(first["snapshot"], second["snapshot"])

    def test_large_workspace_rejected_before_any_content_is_read(self):
        (self.workspace / "accidental-dataset.bin").write_bytes(b"123456")
        self.store.snapshot_limit = 5
        with patch.object(remote, "digest") as hashing:
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                self.store.snapshot("demo")
            hashing.assert_not_called()
        self.assertEqual(self.drive.upload_names(), [])

    def restored_text(self, name):
        replacement = self.replacement(f"vm-{len(list(self.base.iterdir()))}")
        replacement.prepare("demo")
        return (replacement.workspace("demo") / name).read_text()

    def test_a_file_rewritten_while_hashing_is_archived_whole(self):
        (self.workspace / "data").write_text("stable")
        real = remote.digest

        def changing(path):
            if path.name == "data":
                path.write_text("changed by a live writer")
            return real(path)
        with patch.object(remote, "digest", side_effect=changing):
            saved = self.store.snapshot("demo")
        self.assertEqual(saved["files"]["data"]["sha256"], hashlib.sha256(b"changed by a live writer").hexdigest())
        self.assertEqual(self.restored_text("data"), "changed by a live writer")


class LiveWriterTests(StoreCase):
    """Found live: a script that rewrites its progress file every second failed most periodic saves."""

    def setUp(self):
        super().setUp()
        self.start = patch.object(remote.time, "sleep")
        self.sleep = self.start.start()
        self.addCleanup(self.start.stop)

    def restored_text(self, name):
        replacement = self.replacement("vm-restore")
        replacement.prepare("demo")
        return (replacement.workspace("demo") / name).read_text()

    def test_a_file_replaced_atomically_during_the_save_is_archived_whole(self):
        progress = self.workspace / "outputs" / "progress.json"
        progress.write_text('{"completed": 1}')
        real = remote.inventory

        def then_replace(*args, **kwargs):
            listed = real(*args, **kwargs)
            temporary = progress.with_suffix(".tmp")
            temporary.write_text('{"completed": 2}')
            os.replace(temporary, progress)  # How training scripts publish a checkpoint.
            return listed
        with patch.object(remote, "inventory", side_effect=then_replace):
            saved = self.store.snapshot("demo")
        self.assertEqual(saved["files"]["outputs/progress.json"]["sha256"],
                         hashlib.sha256(b'{"completed": 2}').hexdigest())
        self.assertEqual(self.restored_text("outputs/progress.json"), '{"completed": 2}')

    def test_a_file_appended_while_it_is_read_is_retried_and_never_torn(self):
        log = self.workspace / "outputs" / "run.log"
        log.write_text("step 1\n")
        real_read = remote._HashingReader.read
        appended = []

        def read(reader, size=-1):
            data = real_read(reader, size)
            if not appended and data:
                appended.append(True)
                with log.open("a") as stream:
                    stream.write("step 2\n")
            return data
        with patch.object(remote._HashingReader, "read", read):
            saved = self.store.snapshot("demo")
        self.assertEqual(self.sleep.call_count, 1)  # One short, random pause before the retry.
        self.assertEqual(saved["files"]["outputs/run.log"]["sha256"], hashlib.sha256(b"step 1\nstep 2\n").hexdigest())
        self.assertEqual(self.restored_text("outputs/run.log"), "step 1\nstep 2\n")

    def test_a_file_that_never_stops_changing_is_refused_not_archived_torn(self):
        log = self.workspace / "outputs" / "run.log"
        log.write_text("step 1\n")
        real_read = remote._HashingReader.read

        def read(reader, size=-1):
            data = real_read(reader, size)
            if data:
                with log.open("a") as stream:
                    stream.write("more\n")
            return data
        with patch.object(remote._HashingReader, "read", read):
            with self.assertRaisesRegex(RuntimeError, "File changed during checkpoint: outputs/run.log"):
                self.store.snapshot("demo")
        self.assertEqual(self.drive.upload_names(), [])

    def test_a_busy_log_never_fails_because_a_large_file_was_hashed_first(self):
        (self.workspace / "big.bin").write_bytes(b"x" * 1000)
        log = self.workspace / "outputs" / "run.log"
        log.write_text("step 1\n")
        real = remote.digest

        def hashing(path):
            if path.name == "big.bin":
                with log.open("a") as stream:
                    stream.write("step 2\n")  # Written while the big file hashes.
            return real(path)
        with patch.object(remote, "digest", side_effect=hashing):
            listed = remote.inventory(self.workspace)
        self.assertEqual(listed["outputs/run.log"]["size"], len("step 1\nstep 2\n"))

    def test_a_temporary_file_removed_while_listing_is_skipped(self):
        (self.workspace / "data").write_text("kept")
        real_walk = os.walk

        def walk(root, **kwargs):
            for directory, names, files in real_walk(root, **kwargs):
                yield directory, names, files + (["gone.lock"] if Path(directory) == self.workspace else [])
        with patch.object(remote.os, "walk", walk):
            listed = remote.listed_files(self.workspace)
        self.assertIn("data", listed)
        self.assertNotIn("gone.lock", listed)

    def test_snapshot_preserves_previous_generation_when_disk_is_full(self):
        first = self.store.snapshot("demo")
        (self.workspace / "changed").write_text("new")
        with patch.object(remote.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient scratch"):
                self.store.snapshot("demo")
        self.assertEqual(self.store.snapshots("demo")[0]["snapshot"], first["snapshot"])
        self.assertEqual(len(self.drive.upload_names()), 2)

    def test_restore_checks_space_before_downloading(self):
        (self.workspace / "data").write_text("important")
        self.store.snapshot("demo")
        replacement = self.replacement()
        with patch.object(remote.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
             patch.object(self.drive, "download", side_effect=AssertionError("must not download")), \
             patch.object(remote, "digest", side_effect=AssertionError("must not hash")):
            with self.assertRaisesRegex(RuntimeError, "Insufficient scratch"):
                replacement.restore("demo")
        self.assertFalse(replacement.workspace("demo").exists())

    def test_restore_refuses_checkpoint_above_the_size_limit_before_downloading(self):
        (self.workspace / "data").write_bytes(b"x" * 100)
        self.store.snapshot("demo")
        replacement = self.replacement(snapshot_limit=10)
        with patch.object(self.drive, "download", side_effect=AssertionError("must not download")):
            with self.assertRaisesRegex(RuntimeError, "exceeds the configured snapshot size limit"):
                replacement.restore("demo")

    def test_corrupted_archive_bytes_fail_the_sha256_check(self):
        (self.workspace / "data").write_text("important")
        manifest = self.store.snapshot("demo")
        original = self.drive.data_of(manifest["archive_file_id"])
        self.drive.corrupt(manifest["archive_file_id"], flipped(original), keep_metadata=True)
        replacement = self.replacement()
        with self.assertRaisesRegex(drive_api.DriveError, "SHA-256"):
            replacement.restore("demo")
        self.assertFalse(replacement.workspace("demo").exists())
        self.assertEqual(list((self.base / "vm2" / "demo").iterdir()), [])  # Temporary download removed.

    def test_archive_whose_drive_checksum_differs_is_never_chosen(self):
        (self.workspace / "data").write_text("important")
        manifest = self.store.snapshot("demo")
        self.drive.corrupt(manifest["archive_file_id"], b"corrupt")
        replacement = self.replacement()
        with self.assertRaisesRegex(RuntimeError, "no matching archive"):
            replacement.latest("demo")
        with self.assertRaisesRegex(RuntimeError, "no matching archive"):
            replacement.prepare("demo")
        self.assertFalse(replacement.workspace("demo").exists())
        self.assertEqual(replacement.snapshots("demo"), [])

    def test_tampered_manifest_cache_is_fetched_again(self):
        (self.workspace / "data").write_text("important")
        manifest = self.store.snapshot("demo")
        cache = self.store._manifest_cache(manifest["manifest_file_id"])
        cache.write_text(json.dumps({"schema": 1, "snapshot": "tampered"}))
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], manifest["snapshot"])

    def test_restore_never_overwrites_existing_work(self):
        self.store.snapshot("demo")
        (self.workspace / "precious").write_text("uncommitted")
        with self.assertRaisesRegex(RuntimeError, "never overwritten"):
            self.store.restore("demo")
        self.assertEqual((self.workspace / "precious").read_text(), "uncommitted")

    def test_restore_of_an_explicit_snapshot_and_of_an_unknown_one(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        (self.workspace / "data").write_text("v2")
        self.store.snapshot("demo")
        replacement = self.replacement()
        result = replacement.restore("demo", snapshot=first["snapshot"])
        self.assertEqual(result["snapshot"], first["snapshot"])
        self.assertEqual((replacement.workspace("demo") / "data").read_text(), "v1")
        other = self.replacement("vm3")
        with self.assertRaisesRegex(RuntimeError, "No completed snapshot"):
            other.restore("demo", snapshot="20000101T000000000000Z-00000000")

    def test_restore_without_any_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "No completed snapshot"):
            self.replacement().restore("demo")

    def test_credentials_and_links_are_excluded(self):
        (self.workspace / ".env").write_text("SECRET=example")
        (self.workspace / ".env.local").write_text("SECRET=example")
        (self.workspace / ".ssh").mkdir()
        (self.workspace / ".ssh" / "private").write_text("sensitive")
        (self.workspace / ".git").mkdir()
        (self.workspace / ".git" / "config").write_text("[remote]")
        (self.workspace / ".venv").mkdir()
        (self.workspace / ".venv" / "lib.py").write_text("cache")
        (self.workspace / "server.pem").write_text("certificate")
        (self.workspace / "id.key").write_text("key")
        (self.workspace / "download.partial").write_text("incomplete")
        outside = self.base / "outside"
        outside.write_text("not part of project")
        (self.workspace / "linked").symlink_to(outside)
        (self.workspace / "linked-dir").symlink_to(self.base)
        (self.workspace / "valid.py").write_text("print(1)")
        manifest = self.store.snapshot("demo")
        self.assertEqual(set(manifest["files"]), {"valid.py"})
        archive = self.drive.data_of(manifest["archive_file_id"])
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            self.assertEqual(source.getnames(), ["valid.py"])

    def test_malicious_archive_cannot_escape_destination(self):
        cases = {"traversal": [file_member("../escaped", b"evil")],
                 "absolute": [file_member(str(self.base / "absolute-escape"), b"evil")]}
        link = tarfile.TarInfo("link")
        link.type, link.linkname = tarfile.SYMTYPE, str(self.base)
        cases["symlink"] = [(link, None)]
        for label, members in cases.items():
            with self.subTest(label=label):
                archive = self.base / f"{label}.tar.gz"
                archive.write_bytes(tar_bytes(members))
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    remote.extract_archive(archive, self.workspace)
        self.assertFalse((self.workspace.parent / "escaped").exists())
        self.assertFalse((self.base / "absolute-escape").exists())
        self.assertFalse((self.workspace / "link").exists())

    def test_malicious_archive_in_drive_is_refused_on_restore(self):
        data = tar_bytes([file_member("../escaped", b"evil")])
        identifier = "29990101T000000000000Z-0badc0de"
        manifest = {"schema": 1, "project": "demo", "snapshot": identifier, "created_at": "",
                    "reason": "attack", "files": {"../escaped": {"size": 4, "sha256": sha256(b"evil"), "mode": 420}},
                    "archive_sha256": sha256(data), "archive_bytes": len(data)}
        self.drive.put(PROJECT, identifier + ".tar.gz", data)
        self.drive.put(PROJECT, identifier + ".json", remote.manifest_bytes(manifest))
        replacement = self.replacement()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            replacement.restore("demo")
        self.assertFalse(replacement.workspace("demo").exists())
        self.assertFalse((self.base / "vm2" / "escaped").exists())

    def test_restored_files_must_match_the_manifest(self):
        data = tar_bytes([file_member("data", b"actual")])
        identifier = "29990101T000000000000Z-0000beef"
        manifest = {"schema": 1, "project": "demo", "snapshot": identifier, "created_at": "",
                    "reason": "test", "files": {"data": {"size": 6, "sha256": sha256(b"claim!"), "mode": 420}},
                    "archive_sha256": sha256(data), "archive_bytes": len(data)}
        self.drive.put(PROJECT, identifier + ".tar.gz", data)
        self.drive.put(PROJECT, identifier + ".json", remote.manifest_bytes(manifest))
        replacement = self.replacement()
        with self.assertRaisesRegex(RuntimeError, "do not match the checkpoint manifest"):
            replacement.restore("demo")
        self.assertFalse(replacement.workspace("demo").exists())

    def test_read_only_and_group_writable_files_restore(self):
        (self.workspace / "read-only.txt").write_text("frozen")
        (self.workspace / "read-only.txt").chmod(0o444)
        (self.workspace / "shared.txt").write_text("group")
        (self.workspace / "shared.txt").chmod(0o664)
        self.store.snapshot("demo")
        replacement = self.replacement()
        replacement.restore("demo")
        self.assertEqual((replacement.workspace("demo") / "read-only.txt").read_text(), "frozen")


class CommitOrderTests(StoreCase):
    def test_manifest_is_uploaded_only_after_the_archive(self):
        (self.workspace / "data").write_text("v1")
        manifest = self.store.snapshot("demo")
        self.assertEqual(self.drive.upload_names(), [manifest["snapshot"] + ".tar.gz", manifest["snapshot"] + ".json"])
        archive = self.drive.by_name(manifest["snapshot"] + ".tar.gz")
        self.assertEqual(archive["appProperties"]["cp_kind"], "archive")
        self.assertEqual(archive["sha256Checksum"], manifest["archive_sha256"])
        published = self.drive.by_name(manifest["snapshot"] + ".json")
        self.assertEqual(published["data"], remote.manifest_bytes({k: manifest[k] for k in (
            "schema", "project", "snapshot", "created_at", "reason", "files", "archive_sha256", "archive_bytes")}))
        self.assertEqual(published["appProperties"]["cp_kind"], "manifest")
        self.assertEqual(published["appProperties"]["cp_file_count"], "1")

    def test_archive_upload_failure_publishes_no_manifest(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        (self.workspace / "data").write_text("v2")
        last_upload = (self.base / "vm1" / "demo" / "last-upload.json").read_text()
        self.drive.fail_upload(".tar.gz", drive_api.DriveError("Drive request failed: connection reset"))
        with self.assertRaisesRegex(drive_api.DriveError, "connection reset"):
            self.store.snapshot("demo")
        self.assertEqual(self.manifests(), [first["snapshot"] + ".json"])
        self.assertEqual(self.archives(), [first["snapshot"] + ".tar.gz"])
        self.assertEqual((self.base / "vm1" / "demo" / "last-upload.json").read_text(), last_upload)
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], first["snapshot"])

    def test_archive_that_fails_drive_verification_publishes_no_manifest(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        (self.workspace / "data").write_text("v2")
        self.drive.corrupt_uploads.add(".tar.gz")
        with self.assertRaisesRegex(drive_api.DriveError, "mismatch"):
            self.store.snapshot("demo")
        self.assertEqual(len(self.archives()), 2)  # The unverified archive is an ignored orphan.
        self.assertEqual(self.manifests(), [first["snapshot"] + ".json"])
        self.assertEqual(self.replacement().latest("demo")["manifest"]["snapshot"], first["snapshot"])

    def test_manifest_upload_failure_leaves_an_ignored_orphan_and_a_retry_commits(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        (self.workspace / "data").write_text("v2")
        self.drive.fail_upload(".json", drive_api.DriveError("Drive request failed: timeout"))
        with self.assertRaises(drive_api.DriveError):
            self.store.snapshot("demo")
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], first["snapshot"])
        retried = self.store.snapshot("demo")
        self.assertFalse(retried["reused"])
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], retried["snapshot"])
        replacement = self.replacement()
        replacement.prepare("demo")
        self.assertEqual((replacement.workspace("demo") / "data").read_text(), "v2")

    def test_orphan_archive_without_manifest_is_ignored(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        self.drive.put(PROJECT, "29991231T235959999999Z-ffffffff.tar.gz", b"orphan")
        self.drive.put(PROJECT, "29991231T235959999999Z-ffffffff.tar.gz.partial", b"interrupted")
        self.assertEqual([e["snapshot"] for e in self.store.entries("demo")], [first["snapshot"]])
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], first["snapshot"])
        self.assertEqual([m["snapshot"] for m in self.store.snapshots("demo")], [first["snapshot"]])

    def test_broken_newest_pair_makes_latest_raise_and_a_new_checkpoint_supersedes_it(self):
        (self.workspace / "data").write_text("v1")
        older = self.store.snapshot("demo")
        (self.workspace / "data").write_text("v2")
        newest = self.store.snapshot("demo")
        self.drive.trash(newest["archive_file_id"])
        with self.assertRaisesRegex(RuntimeError, "Newest checkpoint .* has no matching archive"):
            self.store.latest("demo")
        replacement = self.replacement()
        with self.assertRaisesRegex(RuntimeError, "no matching archive"):
            replacement.restore("demo")
        self.assertEqual([m["snapshot"] for m in replacement.snapshots("demo")], [older["snapshot"]])
        replacement.restore("demo", snapshot=older["snapshot"])
        self.assertEqual((replacement.workspace("demo") / "data").read_text(), "v1")
        # The workspace still matches the broken manifest, but it is never reused.
        fresh = self.store.snapshot("demo")
        self.assertFalse(fresh["reused"])
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], fresh["snapshot"])

    def test_unparseable_newest_manifest_does_not_block_new_checkpoints(self):
        (self.workspace / "data").write_text("v1")
        self.store.snapshot("demo")
        self.drive.put(PROJECT, "29990101T000000000000Z-deadbeef.json", b"{not json")
        (self.workspace / "data").write_text("v2")
        fresh = self.store.snapshot("demo")
        self.assertFalse(fresh["reused"])

    def test_duplicate_project_folders(self):
        (self.workspace / "data").write_text("v1")
        self.store.snapshot("demo")
        self.drive.add_duplicate_folder(PROJECT)
        (self.workspace / "data").write_text("v2")
        # This VM remembers the folder it created and keeps writing there.
        self.store.snapshot("demo")
        self.assertEqual(len(self.drive.upload_names()), 4)
        # A VM without that memory refuses to guess between two same-named folders.
        with self.assertRaisesRegex(drive_api.DuplicateFolder, "Merge or rename"):
            self.replacement().prepare("demo")

    def test_listing_lag_never_creates_a_second_folder(self):
        fresh = self.replacement("vm-lag")
        workspace = fresh.workspace("lagged")
        workspace.mkdir(parents=True)
        (workspace / "a.txt").write_text("one")
        self.drive.lag = True
        fresh.snapshot("lagged")
        (workspace / "a.txt").write_text("two")
        fresh.snapshot("lagged")
        self.drive.settle()
        projects = self.drive.resolve(["Colab-Persist", "projects"])
        self.assertEqual(len(self.drive.children(projects, name="lagged", folders=True)), 1)

    def test_a_replacement_vm_finds_the_newest_pair_through_the_known_ids(self):
        (self.workspace / "data").write_text("v1")
        self.store.snapshot("demo")
        self.drive.lag = True
        (self.workspace / "data").write_text("v2")
        newest = self.store.snapshot("demo")
        known = {"snapshot": newest["snapshot"], "manifest_file_id": newest["manifest_file_id"],
                 "archive_file_id": newest["archive_file_id"]}
        restored = self.replacement("vm-known").prepare("demo", known=known)
        self.assertEqual(restored["restored"]["snapshot"], newest["snapshot"])

    def test_reuse_refuses_a_pair_that_is_no_longer_intact(self):
        (self.workspace / "data").write_text("stable")
        first = self.store.snapshot("demo")
        self.drive.trash(first["manifest_file_id"])
        second = self.store.snapshot("demo")
        self.assertFalse(second["reused"])
        self.assertNotEqual(second["snapshot"], first["snapshot"])

    def test_reuse_requires_drives_sha256(self):
        (self.workspace / "data").write_text("stable")
        self.drive.report_sha256 = False
        first = self.store.snapshot("demo")
        second = self.store.snapshot("demo")
        self.assertFalse(second["reused"])
        self.assertNotEqual(second["snapshot"], first["snapshot"])

    def test_hard_linked_files_are_archived_as_regular_files(self):
        (self.workspace / "one.bin").write_bytes(b"same inode")
        os.link(self.workspace / "one.bin", self.workspace / "two.bin")
        saved = self.store.snapshot("demo")
        restored = self.replacement("vm-links").prepare("demo")
        self.assertEqual(restored["restored"]["snapshot"], saved["snapshot"])
        workspace = self.replacement("vm-links").workspace("demo")
        self.assertEqual((workspace / "two.bin").read_bytes(), b"same inode")

    def test_an_empty_workspace_restores_and_never_supersedes_a_checkpoint(self):
        (self.workspace / "data").write_text("real work")
        saved = self.store.snapshot("demo")
        fresh = self.replacement("vm-empty")
        (fresh.workspace("demo") / "outputs").mkdir(parents=True)  # As a plain `start` leaves it.
        receipt = fresh.seal()
        self.assertEqual(receipt.get("skipped_empty"), ["demo"])
        self.assertTrue(receipt["nothing_to_save"])
        restored = fresh.prepare("demo")
        self.assertEqual(restored["restored"]["snapshot"], saved["snapshot"])

    def test_list_after_write_lag_is_recovered_through_last_upload(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        self.drive.lag = True
        (self.workspace / "data").write_text("v2")
        second = self.store.snapshot("demo")
        self.assertNotIn(second["snapshot"] + ".json", self.manifests())  # The listing lags.
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], second["snapshot"])
        again = self.store.snapshot("demo")
        self.assertTrue(again["reused"])
        self.assertEqual(again["snapshot"], second["snapshot"])
        self.assertEqual(len(self.drive.upload_names()), 4)
        # A different VM only knows what the listing shows until it catches up.
        self.assertEqual(self.replacement().latest("demo")["manifest"]["snapshot"], first["snapshot"])
        self.drive.settle()
        self.assertEqual(self.replacement("vm3").latest("demo")["manifest"]["snapshot"], second["snapshot"])

    def test_trashed_file_named_in_last_upload_is_not_resurrected(self):
        (self.workspace / "data").write_text("v1")
        first = self.store.snapshot("demo")
        (self.workspace / "data").write_text("v2")
        second = self.store.snapshot("demo")
        self.drive.trash(second["manifest_file_id"])
        self.assertEqual(self.store.latest("demo")["manifest"]["snapshot"], first["snapshot"])


class SealTests(StoreCase):
    def test_active_job_blocks_seal(self):
        (self.workspace / "data").write_text("v1")
        with self.store.project_lock("demo"):
            with self.assertRaisesRegex(RuntimeError, "active managed job"):
                self.store.seal()
        self.assertEqual(self.drive.upload_names(), [])

    def test_workspace_change_during_upload_refuses_seal(self):
        (self.workspace / "data").write_text("v1")

        def late_write(name, _meta):
            if name.endswith(".json"):
                (self.workspace / "late-write").write_text("not saved")
        self.drive.after_upload = late_write
        with self.assertRaisesRegex(RuntimeError, "Workspace changed while uploading"):
            self.store.seal()
        self.assertFalse((self.store.scratch / "flush-receipt.json").exists())

    def test_seal_writes_a_verified_receipt(self):
        (self.workspace / "data").write_text("v1")
        receipt = self.store.seal()
        self.assertTrue(receipt["drive_upload_verified"])
        self.assertTrue(receipt["drive_flushed"])
        self.assertEqual(receipt["checkpoints"][0]["durability"], remote.DURABLE)
        self.assertEqual(receipt["checkpoints"][0]["reason"], "safe_stop")
        stored = json.loads((self.store.scratch / "flush-receipt.json").read_text())
        self.assertEqual(stored["checkpoints"][0]["snapshot"], receipt["checkpoints"][0]["snapshot"])

    def test_seal_refuses_when_an_upload_fails(self):
        (self.workspace / "data").write_text("v1")
        self.drive.fail_upload(".tar.gz", drive_api.QuotaExceeded("Drive quota exhausted (storageQuotaExceeded)",
                                                                  403, "storageQuotaExceeded"))
        with self.assertRaises(drive_api.QuotaExceeded):
            self.store.seal()
        self.assertFalse((self.store.scratch / "flush-receipt.json").exists())

    def test_workspace_without_drive_still_refuses(self):
        (self.workspace / "data").write_text("v1")
        self.drive.tokens = drive_api.TokenSource(path=self.base / "missing-token.json")
        with self.assertRaisesRegex(drive_api.TokenExpired, "Drive is not connected"):
            self.store.seal()
        self.assertFalse((self.store.scratch / "flush-receipt.json").exists())

    def test_runtime_without_workspaces_or_drive_has_nothing_to_save(self):
        drive = fakes.FakeDrive()
        drive.tokens.revoked = True
        store = remote.Store(self.base / "empty-vm", drive=drive)
        receipt = store.seal()
        self.assertTrue(receipt["nothing_to_save"])
        self.assertFalse(receipt["drive_upload_verified"])
        self.assertFalse(receipt["drive_flushed"])
        self.assertEqual(receipt["checkpoints"], [])
        self.assertEqual(drive.calls, [])

    def test_status_reports_workspaces_jobs_and_drive(self):
        self.store._write_job("demo", state="finished")
        status = self.store.status()
        self.assertTrue(status["drive_connected"])
        self.assertEqual(status["drive_mode"], "api")
        self.assertEqual(status["drive_root"], "My Drive/Colab-Persist/projects")
        self.assertEqual(status["workspaces"]["demo"]["job"], "finished")
        self.assertFalse(status["workspaces"]["demo"]["active"])
        with self.store.project_lock("demo"):
            self.assertTrue(self.store.status()["workspaces"]["demo"]["active"])
        self.drive.tokens.revoked = True
        self.assertFalse(self.store.status()["drive_connected"])


class LegacyTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.legacy = self.base / "legacy-drive"
        self.legacy_projects = self.legacy / "MyDrive" / "Colab-CUDA" / "projects"
        mounted = patch("colab_persist.remote.os.path.ismount", side_effect=lambda p: Path(p) == self.legacy)
        mounted.start()
        self.addCleanup(mounted.stop)
        self.old_workspace = self.base / "old-vm" / "workspace"
        (self.old_workspace / "outputs").mkdir(parents=True)
        (self.old_workspace / "train.py").write_text("print('v0.3')\n")
        (self.old_workspace / "outputs" / "model.bin").write_bytes(b"weights" * 100)

    def legacy_store(self):
        return remote.Store(self.base / "vm3", drive=self.drive, legacy_mount=self.legacy)

    def test_manifest_written_by_v03_restores(self):
        manifest, archive, path = v03_checkpoint(self.base / "v03", self.old_workspace, "demo",
                                                 "20250601T120000000000Z-0123abcd")
        self.assertEqual(path.read_bytes(), remote.manifest_bytes(manifest))  # Same serialization.
        # v0.3 files carry no Drive appProperties.
        self.drive.put(PROJECT, archive.name, archive.read_bytes())
        self.drive.put(PROJECT, path.name, path.read_bytes(), mime="application/json")
        replacement = self.replacement()
        result = replacement.prepare("demo")
        self.assertEqual(result["restored"]["snapshot"], manifest["snapshot"])
        self.assertEqual(remote.inventory(replacement.workspace("demo")), manifest["files"])
        self.assertEqual((replacement.workspace("demo") / "train.py").read_text(), "print('v0.3')\n")

    def test_import_copies_the_newest_v03_pair_unchanged_and_skips_it_next_time(self):
        older, _, _ = v03_checkpoint(self.legacy_projects / "demo", self.old_workspace, "demo",
                                     "20250501T120000000000Z-00000001")
        (self.old_workspace / "outputs" / "model.bin").write_bytes(b"better" * 100)
        newest, archive, path = v03_checkpoint(self.legacy_projects / "demo", self.old_workspace, "demo",
                                               "20250601T120000000000Z-00000002")
        store = self.legacy_store()
        before = len(self.drive.upload_names())
        report = store.import_legacy()
        self.assertEqual([(i["project"], i["snapshot"]) for i in report["imported"]], [("demo", newest["snapshot"])])
        self.assertEqual(report["skipped"], [])
        self.assertEqual(self.drive.by_name(path.name)["data"], path.read_bytes())
        self.assertEqual(self.drive.by_name(archive.name)["data"], archive.read_bytes())
        self.assertEqual(self.drive.by_name(path.name)["appProperties"]["cp_snapshot"], newest["snapshot"])
        self.assertNotIn(older["snapshot"] + ".json", self.manifests())

        again = store.import_legacy()
        self.assertEqual(again["imported"], [])
        self.assertEqual([(i["project"], i["snapshot"]) for i in again["skipped"]], [("demo", newest["snapshot"])])
        self.assertTrue(again["skipped"][0]["manifest_file_id"] and again["skipped"][0]["archive_file_id"])
        self.assertEqual(len(self.drive.upload_names()), before + 2)

        everything = store.import_legacy(all_snapshots=True)
        self.assertEqual([i["snapshot"] for i in everything["imported"]], [older["snapshot"]])
        replacement = self.replacement()
        replacement.prepare("demo")
        self.assertEqual((replacement.workspace("demo") / "outputs" / "model.bin").read_bytes(), b"better" * 100)

    def test_import_repairs_a_broken_destination_marker(self):
        newest, archive, path = v03_checkpoint(self.legacy_projects / "demo", self.old_workspace, "demo",
                                               "20250601T120000000000Z-00000002")
        store = self.legacy_store()
        imported = store.import_legacy()["imported"][0]
        self.drive.corrupt(imported["manifest_file_id"], b"{not json")
        report = store.import_legacy()
        self.assertEqual(report["skipped"], [])
        self.assertEqual([r["snapshot"] for r in report["repaired"]], [newest["snapshot"]])
        replacement = self.replacement()
        self.assertEqual(replacement.prepare("demo")["restored"]["snapshot"], newest["snapshot"])

    def test_import_requires_the_legacy_mount(self):
        v03_checkpoint(self.legacy_projects / "demo", self.old_workspace, "demo", "20250601T120000000000Z-00000002")
        store = remote.Store(self.base / "vm3", drive=self.drive, legacy_mount=self.base / "not-mounted")
        with self.assertRaisesRegex(RuntimeError, "legacy Drive mount"):
            store.import_legacy()
        self.assertEqual(self.drive.upload_names(), [])

    def test_import_names_missing_projects(self):
        v03_checkpoint(self.legacy_projects / "demo", self.old_workspace, "demo", "20250601T120000000000Z-00000002")
        with self.assertRaisesRegex(RuntimeError, "Legacy projects not found: missing"):
            self.legacy_store().import_legacy(projects=["demo", "missing"])

    def test_import_skips_incomplete_pairs_and_refuses_changed_archives(self):
        directory = self.legacy_projects / "demo"
        manifest, archive, path = v03_checkpoint(directory, self.old_workspace, "demo",
                                                 "20250601T120000000000Z-00000002")
        # Newer, but its archive is truncated: never a candidate.
        broken, broken_archive, _ = v03_checkpoint(directory, self.old_workspace, "demo",
                                                   "20250701T120000000000Z-00000003")
        broken_archive.write_bytes(broken_archive.read_bytes()[:-10])
        (directory / "notes.json").write_text(json.dumps({"schema": 1, "snapshot": "something-else"}))
        report = self.legacy_store().import_legacy()
        self.assertEqual([i["snapshot"] for i in report["imported"]], [manifest["snapshot"]])

        # Same size, different bytes: upload_file's local hash check refuses before publishing.
        other = self.base / "other-legacy"
        projects = other / "MyDrive" / "Colab-CUDA" / "projects" / "second"
        manifest2, archive2, _ = v03_checkpoint(projects, self.old_workspace, "second",
                                                "20250601T120000000000Z-00000004")
        archive2.write_bytes(flipped(archive2.read_bytes()))
        with patch("colab_persist.remote.os.path.ismount", return_value=True):
            store = remote.Store(self.base / "vm4", drive=self.drive, legacy_mount=other)
            with self.assertRaisesRegex(drive_api.DriveError, "changed before upload"):
                store.import_legacy()
        self.assertEqual(self.drive.listing(["Colab-Persist", "projects", "second"]), [])


class DispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.drive = fakes.FakeDrive()
        real = remote.Store
        self.store_patch = patch.object(remote, "Store", side_effect=lambda **kw: real(
            scratch=self.base / "vm", drive=self.drive, **kw))
        self.env = patch.dict(os.environ, {drive_api.TOKEN_ENV: str(self.base / "shm" / "drive-token.json")})

    def run_dispatch(self, request):
        output = io.StringIO()
        with self.store_patch, self.env, patch("sys.stdout", output):
            try:
                result = remote.dispatch(request)
            except Exception as error:  # noqa: BLE001 - the printed line is under test
                result = error
        return result, output.getvalue().strip().splitlines()[-1]

    def test_dispatch_writes_the_pushed_token_privately_and_reports_the_result(self):
        token = {"token": "fake-access-token", "quota_project": fakes.QUOTA_PROJECT,
                 "expires_at": 4102444800, "folder": "Colab-Persist"}
        result, line = self.run_dispatch({"operation": "prepare", "project": "demo", "drive": token,
                                          "folder": "Colab-Persist"})
        self.assertTrue(line.startswith(remote.RESULT_PREFIX))
        self.assertEqual(json.loads(line[len(remote.RESULT_PREFIX):]), result)
        written = self.base / "shm" / "drive-token.json"
        self.assertEqual(json.loads(written.read_text()), token)
        self.assertEqual(stat.S_IMODE(written.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(written.parent.stat().st_mode), 0o700)

    def test_unsupported_operation_is_reported(self):
        result, line = self.run_dispatch({"operation": "rm_rf", "project": "demo"})
        self.assertIsInstance(result, ValueError)
        self.assertTrue(line.startswith(remote.ERROR_PREFIX))
        self.assertEqual(json.loads(line[len(remote.ERROR_PREFIX):])["operation"], "rm_rf")

    def test_snapshot_and_restore_take_the_project_lock(self):
        store = remote.Store(self.base / "vm", drive=self.drive)
        store.prepare("demo")
        with store.project_lock("demo"):
            result, line = self.run_dispatch({"operation": "snapshot", "project": "demo"})
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("active managed job", json.loads(line[len(remote.ERROR_PREFIX):])["error"])

    def test_connect_reports_the_drive_account(self):
        result, _ = self.run_dispatch({"operation": "connect"})
        self.assertEqual(result, {"drive_connected": True, "drive_email": fakes.EMAIL})


class LocalSnapshotListingTests(unittest.TestCase):
    """backend.snapshots lists checkpoints from this computer through Drive appProperties."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.drive = fakes.FakeDrive()
        patches = [patch.object(backend, "config", return_value={"drive_folder": "Colab-Persist"}),
                   patch.object(backend, "local_drive", return_value=self.drive)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.store = remote.Store(self.base / "vm", drive=self.drive)
        self.store.prepare("demo")

    def test_lists_complete_checkpoints_from_properties_without_downloading(self):
        workspace = self.store.workspace("demo")
        (workspace / "a").write_text("1")
        first = self.store.snapshot("demo")
        (workspace / "b").write_text("2")
        second = self.store.snapshot("demo")
        self.drive.put(PROJECT, "29991231T235959999999Z-ffffffff.tar.gz", b"orphan")
        self.drive.calls.clear()
        listed = backend.snapshots("demo")
        self.assertEqual([s["snapshot"] for s in listed], [second["snapshot"], first["snapshot"]])
        self.assertEqual(listed[0]["file_count"], 2)
        self.assertEqual(listed[0]["archive_sha256"], second["archive_sha256"])
        self.assertEqual(listed[0]["reason"], "manual")
        self.assertNotIn("read_bytes", [op for op, _ in self.drive.calls])
        self.assertNotIn("files", listed[0])

    def test_v03_manifest_without_properties_is_read_and_broken_pairs_are_hidden(self):
        workspace = self.base / "old"
        workspace.mkdir()
        (workspace / "x").write_text("old")
        manifest, archive, path = v03_checkpoint(self.base / "v03", workspace, "demo",
                                                 "20250601T120000000000Z-0123abcd")
        self.drive.put(PROJECT, archive.name, archive.read_bytes())
        self.drive.put(PROJECT, path.name, path.read_bytes())
        missing, _, missing_path = v03_checkpoint(self.base / "v03b", workspace, "demo",
                                                  "20250701T120000000000Z-0badf00d")
        self.drive.put(PROJECT, missing_path.name, missing_path.read_bytes())  # Archive never arrived.
        listed = backend.snapshots("demo")
        self.assertEqual([s["snapshot"] for s in listed], [manifest["snapshot"]])
        self.assertEqual(listed[0]["file_count"], 1)

    def test_missing_project_folder_lists_nothing(self):
        self.assertEqual(backend.snapshots("never-saved"), [])


class EngineDeploymentTests(unittest.TestCase):
    def test_failed_source_size_check_never_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "local-source"
            source.mkdir()
            (source / "train.py").write_text("pass")
            (source / "large.data").write_bytes(b"123456")
            with patch.object(backend, "config", return_value={"snapshot_limit_bytes": 5}), \
                 patch.object(backend.subprocess, "run") as command:
                with self.assertRaisesRegex(RuntimeError, "size limit"):
                    backend.upload_source(object(), "demo", source / "train.py", source)
                command.assert_not_called()

    def test_deployed_engine_can_import_drive_dataset_and_checkpoint_helpers(self):
        calls = []
        with patch.object(backend, "config", return_value={"drive_folder": "Colab-Persist"}), \
             patch.object(backend, "ssh_command", return_value=["unused"]), \
             patch.object(backend, "checked", side_effect=lambda args, **kw: calls.append((args, kw.get("data")))):
            backend.install_engine(object())
        setup_script = calls[1][1].decode()
        self.assertIn(f'export {drive_api.TOKEN_ENV}="{drive_api.DEFAULT_TOKEN_FILE}"', setup_script)
        self.assertIn("gdrive://Colab-Persist/datasets", setup_script)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()
            with tarfile.open(fileobj=io.BytesIO(calls[0][1])) as archive:
                self.assertEqual(sorted(archive.getnames()),
                                 sorted("colab_persist/" + n for n in backend.ENGINE_FILES))
                archive.extractall(target, filter="data")
            program = (f"import sys; sys.path.insert(0, {str(target)!r}); "
                       "from colab_persist.datasets import ShardCache; "
                       "from colab_persist.checkpoints import atomic_checkpoint; "
                       "from colab_persist import drive_api, remote; print(remote.__file__)")
            result = backend.subprocess.run([sys.executable, "-I", "-c", program],
                                            capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(target / "colab_persist" / "remote.py"))


if __name__ == "__main__":
    unittest.main()
