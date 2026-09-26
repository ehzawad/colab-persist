import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from colab_persist import datasets


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # macOS /var is a system symlink; use its explicit canonical target.
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "drive-datasets"
        self.source.mkdir()
        self.cache = self.base / "cache"

    def manifest(self, contents, name="sample"):
        rows = []
        for path, data in contents.items():
            source = self.source / path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(data)
            rows.append({"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return {"schema": 1, "name": name, "version": "v1", "shards": rows}

    def open_cache(self, manifest, budget=16, reserve=0):
        return datasets.ShardCache(manifest, dataset_root=self.source, cache_root=self.cache,
                                   cache_bytes=budget, reserve_bytes=reserve)

    def test_plan_accounts_for_unique_blobs_and_has_stable_fingerprint(self):
        manifest = self.manifest({"a.parquet": b"same", "b.parquet": b"same"})
        result = datasets.plan(manifest, 4, 0)
        self.assertEqual((result["total_bytes"], result["unique_bytes"], result["shard_count"]), (8, 4, 2))
        self.assertEqual(result["largest_shard_bytes"], 4)
        path = self.base / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        self.assertEqual(result, datasets.plan(path, 4, 0))
        manifest["version"] = "v2"
        self.assertNotEqual(result["manifest_sha256"], datasets.plan(manifest, 4, 0)["manifest_sha256"])

    def test_oversized_shard_and_invalid_budgets_fail(self):
        manifest = self.manifest({"shard": b"abcde"})
        for budget in (True, 0, -1, 1.5, 4):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                datasets.plan(manifest, budget, 0)
        for reserve in (True, -1, 1.5):
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                datasets.plan(manifest, 8, reserve)

    def test_rejects_traversal_and_noncanonical_paths(self):
        manifest = self.manifest({"valid": b"data"})
        for path in ("", "../outside", "/absolute", "a/../b", "a//b", "./a", "a/", "a\\b", "C:/a", "a\x00b"):
            manifest["shards"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                datasets.load_manifest(manifest)

    def test_duplicate_paths_and_conflicting_hash_sizes_rejected(self):
        manifest = self.manifest({"a": b"1234"})
        manifest["shards"].append(dict(manifest["shards"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate shard"):
            datasets.load_manifest(manifest)
        manifest["shards"][1].update(path="b", size=9)
        with self.assertRaisesRegex(ValueError, "different sizes"):
            datasets.load_manifest(manifest)

    def test_manifest_limits_duplicate_keys_and_symlink_rejection(self):
        path = self.base / "manifest.json"
        path.write_text('{"schema":1,"schema":1}')
        with self.assertRaisesRegex(ValueError, "Duplicate manifest key"):
            datasets.load_manifest(path)
        with path.open("wb") as handle:
            handle.truncate(datasets.MAX_MANIFEST_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "16 MiB"):
            datasets.load_manifest(path)
        link = self.base / "linked.json"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            datasets.load_manifest(link)
        manifest = self.manifest({"a": b"x"})
        manifest["shards"] *= datasets.MAX_SHARDS + 1
        with self.assertRaisesRegex(ValueError, "shards"):
            datasets.load_manifest(manifest)

    def test_cached_shard_is_readonly_and_reused_without_drive_access(self):
        manifest = self.manifest({"dataset/shard.parquet": b"training-data"})
        cache = self.open_cache(manifest)
        with cache.lease(0) as local:
            self.assertEqual(local.read_bytes(), b"training-data")
            self.assertEqual(local.stat().st_mode & 0o222, 0)
        with patch.object(datasets, "_source_file", side_effect=AssertionError("Drive must not be reopened")):
            with cache.lease("dataset/shard.parquet") as reused:
                self.assertEqual(reused, local)
                self.assertEqual(reused.read_bytes(), b"training-data")
        self.assertEqual((self.source / "dataset/shard.parquet").read_bytes(), b"training-data")

    def test_lru_eviction_is_bounded_and_never_deletes_source(self):
        manifest = self.manifest({"a": b"aaaa", "b": b"bbbb", "c": b"cccc"})
        cache = self.open_cache(manifest, budget=8)
        with cache.lease(0) as a:
            pass
        with cache.lease(1) as b:
            pass
        os.utime(a, ns=(1, 1))
        os.utime(b, ns=(2, 2))
        with cache.lease(2) as c:
            self.assertEqual(c.read_bytes(), b"cccc")
        self.assertFalse(a.exists())
        self.assertTrue(b.exists())
        self.assertEqual(sum(p.stat().st_size for p in cache.blobs.iterdir()), 8)
        self.assertEqual(sorted(p.name for p in self.source.iterdir()), ["a", "b", "c"])

    def test_active_lease_blocks_eviction_then_releases_after_exception(self):
        manifest = self.manifest({"a": b"aaaa", "b": b"bbbb"})
        cache = self.open_cache(manifest, budget=4)
        with self.assertRaisesRegex(LookupError, "reader failed"):
            with cache.lease(0) as a:
                with self.assertRaisesRegex(RuntimeError, "active shard leases"):
                    with cache.lease(1):
                        self.fail("active blob was evicted")
                self.assertEqual(a.read_bytes(), b"aaaa")
                raise LookupError("reader failed")
        with cache.lease(1) as b:
            self.assertEqual(b.read_bytes(), b"bbbb")
        self.assertFalse(a.exists())

    def test_other_process_cannot_evict_active_lease(self):
        manifest = self.manifest({"a": b"aaaa", "b": b"bbbb"})
        path = self.base / "manifest.json"
        path.write_text(json.dumps(manifest))
        cache = self.open_cache(manifest, budget=4)
        script = """from colab_persist.datasets import ShardCache
import sys
c = ShardCache(sys.argv[1], dataset_root=sys.argv[2], cache_root=sys.argv[3], cache_bytes=4, reserve_bytes=0)
try:
    with c.lease(1):
        raise SystemExit(2)
except RuntimeError as error:
    assert 'active shard leases' in str(error), str(error)
"""
        with cache.lease(0) as local:
            result = subprocess.run([sys.executable, "-c", script, str(path), str(self.source), str(self.cache)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(local.exists())

    def test_shared_datasets_share_one_budget_and_deduplicate_content(self):
        one = self.manifest({"first/a": b"aaaa"}, "one")
        two = self.manifest({"second/b": b"bbbb", "second/a": b"aaaa"}, "two")
        first = self.open_cache(one, budget=4)
        second = self.open_cache(two, budget=4)
        with first.lease(0) as a:
            with second.lease(1) as same:
                self.assertEqual(a, same)
            with self.assertRaisesRegex(RuntimeError, "active shard"):
                with second.lease(0):
                    pass
        with second.lease(0):
            pass
        self.assertFalse(a.exists())
        with self.assertRaisesRegex(ValueError, "budget differs"):
            self.open_cache(two, budget=8)

    def test_low_disk_fails_before_source_read(self):
        manifest = self.manifest({"a": b"aaaa"})
        cache = self.open_cache(manifest, reserve=8)
        with patch.object(datasets.shutil, "disk_usage", return_value=SimpleNamespace(free=11)), \
             patch.object(datasets, "_source_file", side_effect=AssertionError("must preflight first")):
            with self.assertRaisesRegex(RuntimeError, "Insufficient local disk"):
                with cache.lease(0):
                    pass

    def test_cache_cannot_overlap_source_data(self):
        manifest = self.manifest({"a": b"aaaa"})
        for source, cache in ((self.source, self.source), (self.source, self.source / "cache"),
                              (self.cache / "blobs", self.cache)):
            with self.subTest(source=source, cache=cache), self.assertRaisesRegex(ValueError, "non-overlapping"):
                datasets.ShardCache(manifest, dataset_root=source, cache_root=cache,
                                    cache_bytes=16, reserve_bytes=0)
        self.assertEqual((self.source / "a").read_bytes(), b"aaaa")

    def test_source_integrity_failure_publishes_nothing(self):
        manifest = self.manifest({"a": b"aaaa"})
        cache = self.open_cache(manifest)
        (self.source / "a").write_bytes(b"evil")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            with cache.lease(0):
                pass
        self.assertEqual(list(cache.blobs.iterdir()), [])
        self.assertEqual((self.source / "a").read_bytes(), b"evil")

    def test_corrupt_cached_blob_is_never_returned(self):
        manifest = self.manifest({"a": b"aaaa"})
        cache = self.open_cache(manifest)
        with cache.lease(0) as local:
            pass
        local.chmod(0o600)
        local.write_bytes(b"evil")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            with cache.lease(0):
                self.fail("corrupt cached blob returned")

    def test_stale_partial_restarts_from_source_and_is_never_leased(self):
        manifest = self.manifest({"a": b"correct"})
        cache = self.open_cache(manifest)
        partial = cache.blobs / (manifest["shards"][0]["sha256"] + ".partial")
        partial.write_bytes(b"incorrect prefix")
        with cache.lease(0) as local:
            self.assertEqual(local.read_bytes(), b"correct")
        self.assertFalse(partial.exists())

    def test_interrupted_copy_and_source_mutation_leave_no_valid_blob(self):
        manifest = self.manifest({"a": b"aaaaaaaa"})
        cache = self.open_cache(manifest)
        original = datasets._source_file
        source_path = self.source / "a"

        @contextlib.contextmanager
        def interrupted(root, relative):
            with original(root, relative) as source:
                class Reader:
                    count = 0

                    def fileno(self):
                        return source.fileno()

                    def read(self, count):
                        self.count += 1
                        if self.count == 2:
                            raise KeyboardInterrupt("simulated interrupted transfer")
                        return source.read(count)
                yield Reader()

        with patch.object(datasets, "_BLOCK_BYTES", 4), patch.object(datasets, "_source_file", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                with cache.lease(0):
                    pass
        self.assertEqual(list(cache.blobs.iterdir()), [])

        @contextlib.contextmanager
        def changed(root, relative):
            with original(root, relative) as source:
                class Reader:
                    changed = False

                    def fileno(self):
                        return source.fileno()

                    def read(self, count):
                        block = source.read(count)
                        if not self.changed:
                            self.changed = True
                            os.utime(source_path, ns=(1, 1))
                        return block
                yield Reader()

        with patch.object(datasets, "_source_file", changed):
            with self.assertRaisesRegex(RuntimeError, "changed during transfer"):
                with cache.lease(0):
                    pass
        self.assertEqual(list(cache.blobs.iterdir()), [])

    def test_symlink_source_file_source_directory_and_cache_are_rejected(self):
        manifest = self.manifest({"nested/a": b"aaaa"})
        cache = self.open_cache(manifest)
        external = self.base / "external"
        external.write_bytes(b"aaaa")
        source = self.source / "nested/a"
        source.unlink()
        source.symlink_to(external)
        with self.assertRaises(OSError):
            with cache.lease(0):
                pass
        source.unlink()
        (self.source / "nested").rmdir()
        (self.source / "nested").symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(OSError):
            with cache.lease(0):
                pass
        alias = self.base / "cache-alias"
        alias.symlink_to(self.cache, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            datasets.ShardCache(manifest, dataset_root=self.source, cache_root=alias,
                                cache_bytes=16, reserve_bytes=0)

    def test_symlink_and_hardlink_cached_entries_are_never_read_or_deleted(self):
        manifest = self.manifest({"a": b"aaaa"})
        cache = self.open_cache(manifest)
        external = self.base / "external"
        external.write_bytes(b"precious")
        blob = cache.blobs / manifest["shards"][0]["sha256"]
        blob.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "Unsafe entry"):
            with cache.lease(0):
                pass
        blob.unlink()
        os.link(external, blob)
        with self.assertRaisesRegex(ValueError, "Unsafe entry"):
            with cache.lease(0):
                pass
        self.assertEqual(external.read_bytes(), b"precious")


if __name__ == "__main__":
    unittest.main()
