"""Read-only Drive shards in a bounded local cache (stdlib only).

The manifest paths are relative to the dataset root, not to the manifest file.
Hold ``cache.lease(shard)`` open for the entire time a reader uses its file.
Transfers are serialized across cache instances. Interrupted copies restart;
only verified, atomically published blobs can be leased. This module never
modifies source shards, and it does not implement prefetch or training resume.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Mapping

DEFAULT_DATASET_ROOT = "/content/drive/MyDrive/Colab-CUDA/datasets"
DEFAULT_CACHE_ROOT = "/content/colab-persist-cache/datasets"
DEFAULT_CACHE_BYTES = 40 * 1024**3
DEFAULT_RESERVE_BYTES = 20 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_SHARDS = 100_000
_HASH = re.compile(r"[a-f0-9]{64}")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BLOCK_BYTES = 4 * 1024**2


@dataclass(frozen=True)
class Shard:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    name: str
    version: str
    shards: tuple[Shard, ...]


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _regular_open(path, flags=os.O_RDONLY, mode=0o600):
    """O_NONBLOCK avoids hanging if an unexpected FIFO is encountered."""
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"Expected a regular file: {path}")
    return fd


def _directory(path, *, create=False):
    """Reject symbolic links in roots and every existing ancestor."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if create:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Directory roots must not contain symlinks: {current}")
    return path


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def load_manifest(source: str | Path | Mapping | Manifest) -> Manifest:
    """Load and validate a schema-1 manifest without touching dataset files."""
    if isinstance(source, Manifest):
        source = {"schema": 1, "name": source.name, "version": source.version,
                  "shards": [vars(shard) for shard in source.shards]}
    if not isinstance(source, Mapping):
        path = Path(source)
        _directory(path.absolute().parent)
        with os.fdopen(_regular_open(path), "rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest exceeds the 16 MiB limit.")
        source = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(source, Mapping) or set(source) != {"schema", "name", "version", "shards"}:
        raise ValueError("Manifest must contain exactly schema, name, version, shards.")
    if type(source["schema"]) is not int or source["schema"] != 1:
        raise ValueError("Only manifest schema 1 is supported.")
    for label in ("name", "version"):
        if not isinstance(source[label], str) or not _LABEL.fullmatch(source[label]):
            raise ValueError(f"Manifest {label} must be a 1-128 character label.")
    rows = source["shards"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_SHARDS:
        raise ValueError(f"Manifest requires 1-{MAX_SHARDS} shards.")
    paths, hashes, shards = set(), {}, []
    serialized_bytes = len(json.dumps({"schema": 1, "name": source["name"],
                                      "version": source["version"], "shards": []},
                                     separators=(",", ":")).encode())
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "size", "sha256"}:
            raise ValueError("Each shard must contain exactly path, size, sha256.")
        path = row["path"]
        if (not isinstance(path, str) or not path or len(path.encode("utf-8")) > 4096
                or "\\" in path or "\x00" in path or any(ord(c) < 32 for c in path)
                or any(p in ("", ".", "..") for p in path.split("/"))
                or PurePosixPath(path).is_absolute() or ":" in path.split("/")[0]):
            raise ValueError("Shard paths must be canonical relative POSIX paths without traversal.")
        if path in paths:
            raise ValueError(f"Duplicate shard path: {path}")
        size = _integer(row["size"], "Shard size")
        sha = row["sha256"]
        if not isinstance(sha, str) or not _HASH.fullmatch(sha):
            raise ValueError("Shard sha256 must be 64 lowercase hex characters.")
        if sha in hashes and hashes[sha] != size:
            raise ValueError("The same sha256 cannot have different sizes.")
        serialized_bytes += len(json.dumps(dict(row), separators=(",", ":")).encode()) + bool(shards)
        if serialized_bytes > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest exceeds the 16 MiB limit.")
        paths.add(path)
        hashes[sha] = size
        shards.append(Shard(path, size, sha))
    return Manifest(source["name"], source["version"], tuple(shards))


def plan(manifest, cache_bytes=DEFAULT_CACHE_BYTES, reserve_bytes=DEFAULT_RESERVE_BYTES):
    manifest = load_manifest(manifest)
    _integer(cache_bytes, "cache_bytes", 1)
    _integer(reserve_bytes, "reserve_bytes")
    largest = max(shard.size for shard in manifest.shards)
    if largest > cache_bytes:
        raise ValueError(f"Largest shard ({largest} bytes) exceeds the cache budget ({cache_bytes}).")
    serialized = {"schema": 1, "name": manifest.name, "version": manifest.version,
                  "shards": [vars(shard) for shard in manifest.shards]}
    fingerprint = hashlib.sha256(json.dumps(serialized, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=True).encode()).hexdigest()
    return {"schema": 1, "name": manifest.name, "version": manifest.version,
            "manifest_sha256": fingerprint,
            "shard_count": len(manifest.shards), "total_bytes": sum(s.size for s in manifest.shards),
            "unique_bytes": sum({s.sha256: s.size for s in manifest.shards}.values()),
            "largest_shard_bytes": largest, "cache_bytes": cache_bytes,
            "reserve_bytes": reserve_bytes, "transfer_policy": "on-demand shard copies; interrupted copies restart",
            "source_policy": "read-only; SHA256 checked against immutable manifest"}


@contextlib.contextmanager
def _source_file(root, relative):
    root = _directory(root)
    folder = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            nested = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=folder)
            os.close(folder)
            folder = nested
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError(f"Source shard is not a regular file: {relative}")
        with os.fdopen(fd, "rb") as handle:
            yield handle
    finally:
        os.close(folder)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class ShardCache:
    """A content-addressed cache; leases protect blobs from concurrent eviction.

    All clients sharing a cache must use the same byte budget. Different
    manifests share that budget and reuse equal SHA256 blobs. Cached files are
    read-only and verified locally on every lease. Readers must not modify them.
    """

    def __init__(self, manifest, *, dataset_root=DEFAULT_DATASET_ROOT,
                 cache_root=DEFAULT_CACHE_ROOT, cache_bytes=DEFAULT_CACHE_BYTES,
                 reserve_bytes=DEFAULT_RESERVE_BYTES):
        self.manifest = load_manifest(manifest)
        plan(self.manifest, cache_bytes, reserve_bytes)
        self.cache_bytes, self.reserve_bytes = cache_bytes, reserve_bytes
        self.dataset_root = Path(dataset_root)
        source_location = self.dataset_root.resolve()
        cache_location = Path(cache_root).resolve()
        if source_location.is_relative_to(cache_location) or cache_location.is_relative_to(source_location):
            raise ValueError("Dataset source and disposable cache must be separate, non-overlapping directories.")
        self.root = _directory(cache_root, create=True)
        self.blobs = _directory(self.root / "blobs", create=True)
        self.locks = _directory(self.root / "locks", create=True)
        with self._global_lock():
            budget = self.root / "budget.json"
            try:
                with os.fdopen(_regular_open(budget), "r") as handle:
                    stored = json.load(handle)
            except FileNotFoundError:
                with os.fdopen(_regular_open(budget, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "w") as handle:
                    json.dump({"cache_bytes": cache_bytes}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            else:
                if stored != {"cache_bytes": cache_bytes}:
                    raise ValueError("Shared cache budget differs; use the same cache_bytes or a separate cache root.")

    @contextlib.contextmanager
    def _global_lock(self):
        _directory(self.root)
        fd = _regular_open(self.root / "cache.lock", os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _directory(self.blobs)
            _directory(self.locks)
            yield
        finally:
            os.close(fd)

    def _blob_lock(self, sha):
        return _regular_open(self.locks / (sha + ".lock"), os.O_RDWR | os.O_CREAT)

    def _select(self, shard):
        if type(shard) is int:
            if not 0 <= shard < len(self.manifest.shards):
                raise ValueError("Shard index is out of range.")
            return self.manifest.shards[shard]
        if isinstance(shard, str):
            matches = [s for s in self.manifest.shards if s.path == shard]
            if matches:
                return matches[0]
        elif isinstance(shard, Shard) and shard in self.manifest.shards:
            return shard
        raise ValueError("Shard must belong to this manifest.")

    def _entries(self):
        entries = []
        for path in self.blobs.iterdir():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"Unsafe entry in dataset cache: {path.name}")
            if path.name.endswith(".partial") and _HASH.fullmatch(path.name[:-8]):
                # Only the global-lock holder can transfer, so these are stale.
                path.unlink()
            elif _HASH.fullmatch(path.name):
                entries.append((path, info))
            else:
                raise ValueError(f"Unrecognized entry in dataset cache: {path.name}")
        return entries

    def _room(self, needed, protected):
        entries = self._entries()
        total = sum(info.st_size for _, info in entries)
        for path, info in sorted(entries, key=lambda entry: entry[1].st_mtime_ns):
            free = shutil.disk_usage(self.root).free
            if total + needed <= self.cache_bytes and free >= needed + self.reserve_bytes:
                return
            if path.name == protected:
                continue
            lock = self._blob_lock(path.name)
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                current = path.lstat()
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                    raise ValueError("Unsafe cache entry during eviction.")
                path.unlink()
                total -= info.st_size
            finally:
                os.close(lock)
        if total + needed > self.cache_bytes:
            raise RuntimeError("Cache budget is occupied by active shard leases; release a lease before staging another shard.")
        free = shutil.disk_usage(self.root).free
        if free < needed + self.reserve_bytes:
            raise RuntimeError(f"Insufficient local disk: require {needed} shard bytes plus {self.reserve_bytes} reserve bytes; {free} free.")

    def _verify_cached(self, path, shard):
        with os.fdopen(_regular_open(path), "rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_nlink != 1 or before.st_size != shard.size:
                raise RuntimeError("Cached shard integrity check failed (size or hard link).")
            sha = hashlib.sha256()
            for block in iter(lambda: handle.read(_BLOCK_BYTES), b""):
                sha.update(block)
            if sha.hexdigest() != shard.sha256 or _identity(before) != _identity(os.fstat(handle.fileno())):
                raise RuntimeError("Cached shard checksum failed; cache contents were modified.")

    def _copy(self, shard, destination):
        partial = destination.with_name(destination.name + ".partial")
        try:
            with _source_file(self.dataset_root, shard.path) as source:
                before = os.fstat(source.fileno())
                if before.st_size != shard.size:
                    raise RuntimeError(f"Source shard size differs from manifest: {shard.path}")
                sha, copied = hashlib.sha256(), 0
                with os.fdopen(_regular_open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb") as target:
                    for block in iter(lambda: source.read(_BLOCK_BYTES), b""):
                        copied += len(block)
                        if copied > shard.size:
                            raise RuntimeError("Source shard changed during transfer.")
                        # Other processes may consume space while a transfer runs.
                        if shutil.disk_usage(self.root).free < len(block) + self.reserve_bytes:
                            raise RuntimeError("Insufficient local disk to retain the configured reserve.")
                        target.write(block)
                        sha.update(block)
                    target.flush()
                    os.fsync(target.fileno())
                if _identity(before) != _identity(os.fstat(source.fileno())):
                    raise RuntimeError("Source shard changed during transfer.")
                if copied != shard.size or sha.hexdigest() != shard.sha256:
                    raise RuntimeError(f"Source shard checksum differs from manifest: {shard.path}")
            partial.chmod(0o444)
            partial.replace(destination)
            directory = os.open(self.blobs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            # SIGKILL can leave a partial; the next transfer discards it safely.
            partial.unlink(missing_ok=True)
            raise

    @contextlib.contextmanager
    def lease(self, shard):
        shard = self._select(shard)
        path = self.blobs / shard.sha256
        lock = None
        try:
            with self._global_lock():
                self._entries()
                try:
                    path.lstat()
                except FileNotFoundError:
                    self._room(shard.size, shard.sha256)
                    self._copy(shard, path)
                else:
                    self._verify_cached(path, shard)
                    self._room(0, shard.sha256)
                lock = self._blob_lock(shard.sha256)
                fcntl.flock(lock, fcntl.LOCK_SH)
                os.utime(path, None, follow_symlinks=False)
            yield path
        finally:
            if lock is not None:
                os.close(lock)
