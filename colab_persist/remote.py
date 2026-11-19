"""Dependency-free checkpoint engine, copied to the disposable Linux runtime.

Checkpoints are immutable ``<id>.tar.gz`` archives plus a schema-1 JSON manifest
in ``My Drive/<folder>/projects/<project>/``. They are written directly from the VM
through the Drive API with a short-lived access token that the local client
supplies; Drive's own size and SHA-256 confirm every upload. The manifest is the
commit marker and is uploaded only after its archive verifies.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

try:
    from . import drive_api
except ImportError:  # Executed as a plain file.
    import drive_api  # type: ignore[no-redef]

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".ssh", ".config",
            ".codex", ".claude", ".gemini", "node_modules", ".DS_Store"}
RESULT_PREFIX = "COLAB_PERSIST_RESULT="
ERROR_PREFIX = "COLAB_PERSIST_ERROR="
GIB = 1024 ** 3
DEFAULT_SNAPSHOT_LIMIT = 5 * GIB
DEFAULT_DISK_RESERVE = 2 * GIB
DEFAULT_FOLDER = "Colab-Persist"
LEGACY_FOLDER = "Colab-CUDA"
ENGINE_ROOT = "/content/.colab-persist"
FINAL_SAVE_WAIT_SECONDS = 24 * 3600
DRIVE_RETRY_SECONDS = 30
LOG_CHUNK_BYTES = 64 * 1024
DURABLE = "drive_upload_verified"


def valid_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Project names must be 1-64 letters, digits, underscores or hyphens.")
    return value


def excluded(path: Path) -> bool:
    return any(p in EXCLUDED or p == ".env" or p.startswith(".env.") or
               p.endswith((".pem", ".key", ".partial", ".tmp")) for p in path.parts)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def manifest_bytes(manifest: dict) -> bytes:
    """The exact serialization v0.3 wrote with write_json."""
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory(root: Path, max_bytes: int | None = None) -> dict:
    """Only regular files inside the workspace; never follow symbolic links."""
    result, candidates, total = {}, [], 0
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(n for n in names if not excluded(Path(directory, n).relative_to(root))
                          and not Path(directory, n).is_symlink())
        for name in sorted(files):
            path = Path(directory, name)
            relative = path.relative_to(root)
            if excluded(relative) or path.is_symlink():
                continue
            before = path.stat()
            if not stat.S_ISREG(before.st_mode):
                continue
            total += before.st_size
            if max_bytes is not None and total > max_bytes:
                raise RuntimeError("Workspace/source exceeds the snapshot size limit. Keep datasets and base "
                                   "model caches outside the workspace; use the bounded dataset cache.")
            candidates.append((path, relative, before))
    # Complete the metadata size check before hashing even the first large file.
    for path, relative, before in candidates:
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"File changed during checkpoint: {relative}. Retry after its writer finishes.")
        result[relative.as_posix()] = {"size": before.st_size, "sha256": sha,
                                       "mode": stat.S_IMODE(before.st_mode)}
    return result


def extract_archive(archive: Path, target: Path, max_bytes: int | None = None) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if max_bytes is not None and sum(m.size for m in members) > max_bytes:
            raise RuntimeError("Archive contents exceed the snapshot size limit.")
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe path or special file in checkpoint archive.")
        source.extractall(target, members=members, filter="data")


def summary_properties(manifest: dict) -> dict:
    """Small private Drive properties so listings need not download manifests."""
    return {"cp_kind": "manifest", "cp_snapshot": manifest["snapshot"],
            "cp_created_at": manifest.get("created_at", ""), "cp_reason": str(manifest.get("reason", ""))[:60],
            "cp_archive_bytes": str(manifest.get("archive_bytes", "")),
            "cp_file_count": str(len(manifest.get("files", {}))),
            "cp_archive_sha256": manifest.get("archive_sha256", "")}


def archive_matches(meta: dict | None, manifest: dict) -> bool:
    if not meta:
        return False
    if int(meta.get("size", -1)) != manifest.get("archive_bytes"):
        return False
    stored = meta.get("sha256Checksum")
    return stored is None or stored == manifest["archive_sha256"]


class Store:
    def __init__(self, scratch="/content/colab-persist", folder=DEFAULT_FOLDER,
                 snapshot_limit=DEFAULT_SNAPSHOT_LIMIT, drive=None, legacy_mount="/content/drive"):
        self.scratch = Path(scratch)
        self.folder = valid_name(folder)
        self.legacy_mount = Path(legacy_mount)
        if type(snapshot_limit) is not int or snapshot_limit <= 0:
            raise ValueError("snapshot_limit must be a positive byte count.")
        self.snapshot_limit = snapshot_limit
        self._drive = drive

    # -- Drive ---------------------------------------------------------------------
    @property
    def drive(self):
        if self._drive is None:
            self._drive = drive_api.DriveClient(drive_api.TokenSource.from_environment())
        return self._drive

    def require_drive(self):
        tokens = getattr(self.drive, "tokens", None)
        if tokens is not None:
            tokens.get()  # Raises a clear TokenExpired before any work starts.
        return self.drive

    def project_folder(self, project, create=False):
        parts = [self.folder, "projects", valid_name(project)]
        if not create:
            return self.require_drive().resolve(parts)
        with self._folder_lock():
            return self.require_drive().resolve(parts, create=True)

    @contextlib.contextmanager
    def _folder_lock(self):
        self.scratch.mkdir(parents=True, exist_ok=True)
        with (self.scratch / ".drive-folders.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def workspace(self, project):
        return self.scratch / valid_name(project) / "workspace"

    def _last_upload_path(self, project):
        return self.scratch / valid_name(project) / "last-upload.json"

    def _manifest_cache(self, file_id):
        return self.scratch / ".manifests" / (re.sub(r"[^A-Za-z0-9_-]", "_", file_id) + ".json")

    @contextlib.contextmanager
    def project_lock(self, project):
        base = self.scratch / valid_name(project)
        base.mkdir(parents=True, exist_ok=True)
        with (base / "job.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Project {project} has an active managed job; wait before saving, restoring or stopping.")
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    # -- checkpoint listing ----------------------------------------------------------
    def entries(self, project):
        """Manifest/archive candidates, newest first. Orphan archives are ignored."""
        folder = self.project_folder(project)
        files = self.drive.children(folder, folders=False) if folder else []
        last = self._last_upload_path(project)
        if last.exists():
            # Listings can lag behind a write; a known ID is fetched directly.
            known = json.loads(last.read_text())
            ids = {item["id"] for item in files}
            for key in ("manifest_id", "archive_id"):
                if known.get(key) and known[key] not in ids:
                    item = self.drive.get(known[key])
                    if item and not item.get("trashed"):
                        files.append(item)
        grouped = {}
        for item in files:
            name = item["name"]
            if name.endswith(".json"):
                grouped.setdefault(name[:-5], {"manifests": [], "archives": []})["manifests"].append(item)
            elif name.endswith(".tar.gz"):
                grouped.setdefault(name[:-7], {"manifests": [], "archives": []})["archives"].append(item)
        result = []
        for snapshot in sorted(grouped, reverse=True):
            group = grouped[snapshot]
            if group["manifests"]:
                result.append({"snapshot": snapshot, "manifest": group["manifests"][0],
                               "manifests": group["manifests"], "archives": group["archives"]})
        return result

    def load_manifest(self, entry):
        meta = entry["manifest"]
        cache = self._manifest_cache(meta["id"])
        raw = None
        if cache.exists():
            raw = cache.read_bytes()
            if meta.get("md5Checksum") and hashlib.md5(raw).hexdigest() != meta["md5Checksum"]:
                raw = None
        if raw is None:
            raw = self.drive.read_bytes(meta["id"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        manifest = json.loads(raw)
        if manifest.get("schema") != 1 or manifest.get("snapshot") != entry["snapshot"]:
            raise RuntimeError(f"Checkpoint manifest {entry['snapshot']} is not a valid schema-1 manifest.")
        return manifest

    def archive_for(self, entry, manifest):
        matches = [item for item in entry["archives"] if archive_matches(item, manifest)]
        return matches[0] if matches else None

    def snapshots(self, project):
        result = []
        for entry in self.entries(project):
            try:
                manifest = self.load_manifest(entry)
            except (RuntimeError, ValueError):
                continue
            if self.archive_for(entry, manifest):
                result.append(manifest)
        return result

    def latest(self, project):
        """Newest committed checkpoint, refusing to skip silently past a broken one."""
        entries = self.entries(project)
        if not entries:
            return None
        entry = entries[0]
        manifest = self.load_manifest(entry)
        archive = self.archive_for(entry, manifest)
        if archive is None:
            raise RuntimeError(f"Newest checkpoint {entry['snapshot']} has no matching archive in Drive. "
                               "Restore an older snapshot explicitly with --snapshot.")
        return {"entry": entry, "manifest": manifest, "archive": archive}

    # -- save --------------------------------------------------------------------
    def snapshot(self, project, reason="manual"):
        self.require_drive()
        workspace = self.workspace(project)
        if not workspace.is_dir():
            raise RuntimeError(f"Workspace {project} has not been prepared.")
        files = inventory(workspace, self.snapshot_limit)
        try:
            previous = self.latest(project)
        except drive_api.DriveError:
            raise
        except (RuntimeError, ValueError):
            previous = None  # A broken newest pair is never reused; a fresh checkpoint supersedes it.
        if previous and previous["manifest"]["files"] == files:
            fresh = self.drive.get(previous["archive"]["id"])
            if fresh and not fresh.get("trashed") and archive_matches(fresh, previous["manifest"]):
                return {**previous["manifest"], "reused": True, "durability": DURABLE,
                        "manifest_file_id": previous["entry"]["manifest"]["id"], "archive_file_id": fresh["id"]}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        identifier = stamp + "-" + uuid.uuid4().hex[:8]
        # gzip can grow incompressible input; allow headers plus a free-disk reserve.
        required = int(sum(f["size"] for f in files.values()) * 1.02) + len(files) * 4096
        if shutil.disk_usage(self.scratch).free < required + DEFAULT_DISK_RESERVE:
            raise RuntimeError("Insufficient scratch disk for a workspace archive and the 2 GiB reserve.")
        with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=self.scratch) as temporary:
            archive = Path(temporary) / (identifier + ".tar.gz")
            with tarfile.open(archive, "w:gz", compresslevel=1) as target:
                for name in files:
                    target.add(workspace / name, arcname=name, recursive=False)
            # Validate what was actually archived; a live writer cannot silently produce a good manifest.
            with tarfile.open(archive, "r:gz") as source:
                for member in source.getmembers():
                    h = hashlib.sha256()
                    with source.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                            h.update(block)
                    if h.hexdigest() != files[member.name]["sha256"]:
                        raise RuntimeError(f"File changed during archive creation: {member.name}")
            manifest = {"schema": 1, "project": project, "snapshot": identifier,
                        "created_at": now(), "reason": reason, "files": files,
                        "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size}
            folder = self.project_folder(project, create=True)
            archive_meta = self.drive.upload_file(
                archive, folder, archive.name, sha256=manifest["archive_sha256"], mime="application/gzip",
                app_properties={"cp_kind": "archive", "cp_snapshot": identifier,
                                "cp_sha256": manifest["archive_sha256"]})
        # The manifest is the commit marker, published only after the archive verified.
        data = manifest_bytes(manifest)
        manifest_meta = self.drive.upload_bytes(data, folder, identifier + ".json",
                                                app_properties=summary_properties(manifest))
        cache = self._manifest_cache(manifest_meta["id"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        write_json(self._last_upload_path(project), {"snapshot": identifier, "manifest_id": manifest_meta["id"],
                                                      "archive_id": archive_meta["id"]})
        return {**manifest, "reused": False, "durability": DURABLE,
                "manifest_file_id": manifest_meta["id"], "archive_file_id": archive_meta["id"]}

    # -- restore -------------------------------------------------------------------
    def restore(self, project, snapshot="latest"):
        self.require_drive()
        workspace = self.workspace(project)
        if workspace.exists() and any(workspace.iterdir()):
            raise RuntimeError("Restore requires an empty workspace; existing work is never overwritten.")
        if snapshot == "latest":
            chosen = self.latest(project)
            if chosen is None:
                raise RuntimeError("No completed snapshot found.")
            manifest, archive_meta = chosen["manifest"], chosen["archive"]
        else:
            entry = next((e for e in self.entries(project) if e["snapshot"] == snapshot), None)
            if entry is None:
                raise RuntimeError("No completed snapshot found.")
            manifest = self.load_manifest(entry)
            archive_meta = self.archive_for(entry, manifest)
            if archive_meta is None:
                raise RuntimeError(f"Checkpoint {snapshot} has no matching archive in Drive.")
        restored_size = sum(f["size"] for f in manifest["files"].values())
        if restored_size > self.snapshot_limit:
            raise RuntimeError("Checkpoint exceeds the configured snapshot size limit; review its contents first.")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        needed = restored_size + manifest["archive_bytes"] + DEFAULT_DISK_RESERVE
        if shutil.disk_usage(workspace.parent).free < needed:
            raise RuntimeError("Insufficient scratch disk to download, restore and retain the 2 GiB reserve.")
        with tempfile.TemporaryDirectory(prefix="restore-", dir=workspace.parent) as temporary:
            archive = Path(temporary) / "checkpoint.tar.gz"
            self.drive.download(archive_meta["id"], archive, expected_size=manifest["archive_bytes"],
                                expected_sha256=manifest["archive_sha256"])
            staging = Path(temporary) / "workspace"
            staging.mkdir()
            extract_archive(archive, staging, self.snapshot_limit)
            # The safe "data" extraction filter normalizes permissions; reapply the recorded ones.
            for name, info in manifest["files"].items():
                target = staging / name
                if target.is_file() and not target.is_symlink():
                    os.chmod(target, info.get("mode", 0o644) & 0o777)
            if inventory(staging, self.snapshot_limit) != manifest["files"]:
                raise RuntimeError("Restored files do not match the checkpoint manifest.")
            if workspace.exists():
                workspace.rmdir()  # Empty only; never removes user files.
            staging.rename(workspace)
        return {"project": project, "snapshot": manifest["snapshot"],
                "restored_files": len(manifest["files"]), "workspace": str(workspace),
                "archive_sha256": manifest["archive_sha256"]}

    def prepare(self, project):
        with self.project_lock(project):
            workspace = self.workspace(project)
            restored = None
            if not workspace.exists() or not any(workspace.iterdir()):
                self.require_drive()
                if self.entries(project):
                    restored = self.restore(project)
                else:
                    workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "outputs").mkdir(exist_ok=True)
            return {"workspace": str(workspace), "restored": restored}

    def drive_connected(self):
        try:
            tokens = getattr(self.drive, "tokens", None)
            if tokens is not None:
                tokens.get()
            return True
        except drive_api.DriveError:
            return False

    def status(self):
        jobs = {}
        if self.scratch.exists():
            for base in self.scratch.iterdir():
                if base.is_dir() and (base / "workspace").is_dir():
                    try:
                        with self.project_lock(base.name):
                            active = False
                    except RuntimeError:
                        active = True
                    item = {"active": active, "workspace": str(base / "workspace")}
                    job = self._read_job(base.name)
                    if job:
                        item["job"] = job.get("state")
                    jobs[base.name] = item
        return {"drive_connected": self.drive_connected(), "drive_mode": "api",
                "drive_mounted": os.path.ismount(self.legacy_mount), "workspaces": jobs,
                "drive_root": f"My Drive/{self.folder}/projects"}

    def seal(self):
        """Checkpoint every managed workspace; succeed only when every upload verified.

        With no managed workspace there is nothing to save, so the receipt says so
        instead of refusing the shutdown."""
        projects = list(self.status()["workspaces"])
        if not projects:
            return {"drive_upload_verified": False, "drive_flushed": False, "nothing_to_save": True,
                    "checkpoints": [], "checked_at": now()}
        with contextlib.ExitStack() as stack:
            for project in projects:
                stack.enter_context(self.project_lock(project))
            self.require_drive()
            checkpoints = [self.snapshot(project, "safe_stop") for project in projects]
            if any(inventory(self.workspace(item["project"]), self.snapshot_limit) != item["files"]
                   for item in checkpoints):
                raise RuntimeError("Workspace changed while uploading to Drive. Save again; shutdown refused.")
            stamp = now()
            # drive_flushed is kept for v0.3 callers: it means the data is durably in Drive.
            receipt = {"drive_upload_verified": True, "drive_flushed": True, "checkpoints": checkpoints,
                       "verified_at": stamp, "flushed_at": stamp}
            self.scratch.mkdir(parents=True, exist_ok=True)
            write_json(self.scratch / "flush-receipt.json", receipt)
            return receipt

    # -- detached jobs ---------------------------------------------------------------
    def _job_dir(self, project):
        return self.scratch / valid_name(project)

    def _read_job(self, project):
        path = self._job_dir(project) / "job.json"
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            return None

    def _write_job(self, project, **values):
        state = {**(self._read_job(project) or {}), **values, "updated_at": now()}
        write_json(self._job_dir(project) / "job.json", state)
        return state

    def _validate_script(self, project, script):
        workspace = self.workspace(project)
        relative = PurePosixPath(script)
        if relative.is_absolute() or ".." in relative.parts or not script.endswith(".py"):
            raise ValueError("Script must be a relative Python file inside the workspace.")
        entry = workspace / script
        if entry.is_symlink() or not entry.resolve().is_relative_to(workspace.resolve()) or not entry.is_file():
            raise ValueError("Script is missing or is outside the workspace.")
        return entry

    def launch(self, project, script, arguments, interval=60, run_id=None, stop_after=True):
        self.require_drive()
        if not isinstance(interval, int) or interval < 10:
            raise ValueError("Checkpoint interval must be at least 10 seconds.")
        if not isinstance(arguments, list) or not all(isinstance(a, str) for a in arguments):
            raise ValueError("Script arguments must be a list of strings.")
        self._validate_script(project, script)
        with self.project_lock(project):
            pass  # Refuse early while another managed job holds the project.
        run_id = run_id or uuid.uuid4().hex
        base = self._job_dir(project)
        write_json(base / "job-spec.json", {"run_id": run_id, "project": project, "script": script,
                                            "arguments": arguments, "interval": interval,
                                            "stop_after": bool(stop_after), "folder": self.folder,
                                            "snapshot_limit": self.snapshot_limit, "scratch": str(self.scratch)})
        self._write_job(project, run_id=run_id, state="starting", script=script, arguments=arguments,
                        stop_after=bool(stop_after), exit_code=None, error=None)
        root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
        with (base / "worker.log").open("ab") as log:
            subprocess.Popen([sys.executable, "-m", "colab_persist.remote", "--worker", project,
                              "--scratch", str(self.scratch)],
                             cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True, close_fds=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = self._read_job(project) or {}
            if job.get("run_id") == run_id and job.get("state") != "starting":
                return job
            time.sleep(0.5)
        raise RuntimeError("The job worker did not start; see worker.log in the project directory on the VM.")

    def job_status(self, project, offset=0, tail=None):
        job = self._read_job(project)
        if not job:
            return {"state": "none", "project": project}
        if job.get("state") in ("running", "saving", "awaiting_drive") and job.get("pid"):
            try:
                os.kill(int(job["pid"]), 0)
            except ProcessLookupError:
                job = {**job, "state": "lost",
                       "error": "The job worker stopped unexpectedly. The workspace is still on the VM; "
                                "run `colab-persist save` or `colab-persist stop` to keep it."}
            except PermissionError:
                pass
        log = self.workspace(project) / "outputs" / "run.log"
        chunk, size = "", 0
        if log.exists():
            size = log.stat().st_size
            if tail is not None:
                offset = size - min(int(tail), LOG_CHUNK_BYTES)
            offset = max(0, min(int(offset), size))
            with log.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(LOG_CHUNK_BYTES)
            chunk = data.decode(errors="replace")
            offset += len(data)
        return {**job, "project": project, "log": chunk, "offset": offset, "log_bytes": size}

    def worker(self, project):
        spec = json.loads((self._job_dir(project) / "job-spec.json").read_text())
        try:
            lock = self.project_lock(project)
            lock.__enter__()
        except RuntimeError as error:
            self._write_job(project, state="refused", error=str(error))
            return
        try:
            self._work(project, spec)
        except BaseException as error:
            # Never leave a non-terminal state behind; the local monitor would wait forever.
            self._write_job(project, state="exited_unsaved", error=f"Job worker failed: {error}"[:500])
            raise
        finally:
            lock.__exit__(None, None, None)

    def _work(self, project, spec):
        workspace = self.workspace(project)
        entry = self._validate_script(project, spec["script"])
        log = workspace / "outputs" / "run.log"
        log.parent.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["COLAB_WORKSPACE"] = str(workspace)
        env["COLAB_OUTPUT_DIR"] = str(workspace / "outputs")
        cache = self.scratch.parent / "colab-persist-cache"
        cache.mkdir(parents=True, exist_ok=True)
        env["COLAB_DATASET_ROOT"] = f"gdrive://{self.folder}/datasets"
        env["COLAB_DATASET_CACHE"] = str(cache / "datasets")
        env[drive_api.FOLDER_ENV] = self.folder
        env.setdefault(drive_api.TOKEN_ENV, drive_api.DEFAULT_TOKEN_FILE)
        env.setdefault("HF_HOME", str(cache / "huggingface"))
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        failures = []
        with log.open("ab", buffering=0) as output:
            process = subprocess.Popen([sys.executable, "-u", str(entry), *spec["arguments"]],
                                       cwd=workspace, env=env, stdout=output, stderr=output)
            self._write_job(project, state="running", pid=os.getpid(), script_pid=process.pid,
                            started_at=now(), interval=spec["interval"])

            def terminate(*_):
                process.terminate()
            signal.signal(signal.SIGTERM, terminate)
            last_save = time.monotonic()
            while process.poll() is None:
                time.sleep(1)
                if time.monotonic() - last_save >= spec["interval"]:
                    for attempt in range(3):
                        try:
                            saved = self.snapshot(project, "periodic")
                            self._write_job(project, last_checkpoint=saved["snapshot"], last_checkpoint_at=now(),
                                            drive_waiting=False)
                            break
                        except Exception as error:
                            # A file rewritten mid-archive is retried shortly; it is never saved torn.
                            if "changed during" in str(error) and attempt < 2 and process.poll() is None:
                                time.sleep(1)
                                continue
                            failures.append(str(error))
                            self._write_job(project, last_error=str(error)[:500],
                                            drive_waiting=isinstance(error, drive_api.TokenExpired))
                            break
                    last_save = time.monotonic()
        write_json(workspace / "outputs" / "last-run.json",
                   {"script": spec["script"], "arguments": spec["arguments"], "exit_code": process.returncode,
                    "finished_at": now(), "periodic_checkpoint_errors": failures[-10:]})
        self._write_job(project, state="saving", exit_code=process.returncode,
                        periodic_checkpoint_errors=failures[-10:])
        deadline = time.monotonic() + FINAL_SAVE_WAIT_SECONDS
        while True:
            try:
                saved = self.snapshot(project, "script_exit")
            except (drive_api.QuotaExceeded, ValueError) as error:
                self._write_job(project, state="exited_unsaved", error=str(error)[:500])
                return
            except (drive_api.DriveError, OSError) as error:
                # Token expiry or network loss: keep the job's results and wait for a reconnect.
                if time.monotonic() > deadline:
                    self._write_job(project, state="exited_unsaved", error=str(error)[:500])
                    return
                self._write_job(project, state="awaiting_drive", error=str(error)[:500])
                time.sleep(DRIVE_RETRY_SECONDS)
                continue
            except RuntimeError as error:
                self._write_job(project, state="exited_unsaved", error=str(error)[:500])
                return
            summary = {k: v for k, v in saved.items() if k != "files"}
            summary["file_count"] = len(saved["files"])
            self._write_job(project, state="finished", checkpoint=summary, error=None, finished_at=now())
            return

    # -- one-time import of v0.3 checkpoints --------------------------------------------
    def import_legacy(self, legacy_folder=LEGACY_FOLDER, projects=None, all_snapshots=False):
        root = self.legacy_mount / "MyDrive" / valid_name(legacy_folder) / "projects"
        if not os.path.ismount(self.legacy_mount) or not root.is_dir():
            raise RuntimeError(f"The legacy Drive mount does not show My Drive/{legacy_folder}/projects. "
                               "Run `colab-persist import-legacy` from an interactive terminal.")
        self.require_drive()
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
        if projects:
            missing = sorted(set(projects) - set(names))
            if missing:
                raise RuntimeError(f"Legacy projects not found: {', '.join(missing)}")
            names = [n for n in names if n in projects]
        report = {"imported": [], "skipped": [], "legacy_root": f"My Drive/{legacy_folder}/projects"}
        for project in names:
            valid_name(project)
            manifests = []
            for path in sorted((root / project).glob("*.json"), reverse=True):
                try:
                    raw = path.read_bytes()
                    item = json.loads(raw)
                except (OSError, ValueError):
                    continue
                if item.get("schema") == 1 and item.get("snapshot") == path.stem:
                    archive = root / project / (path.stem + ".tar.gz")
                    if archive.is_file() and archive.stat().st_size == item.get("archive_bytes"):
                        manifests.append((item, raw, archive))
            if not all_snapshots:
                manifests = manifests[:1]
            existing = {entry["snapshot"]: entry for entry in self.entries(project)}
            folder = self.project_folder(project, create=True)
            for manifest, raw, archive in manifests:
                entry = existing.get(manifest["snapshot"])
                if entry and self.archive_for(entry, manifest):
                    report["skipped"].append({"project": project, "snapshot": manifest["snapshot"]})
                    continue
                # upload_file hashes the legacy archive first and refuses a mismatch.
                archive_meta = self.drive.upload_file(
                    archive, folder, archive.name, sha256=manifest["archive_sha256"], mime="application/gzip",
                    app_properties={"cp_kind": "archive", "cp_snapshot": manifest["snapshot"],
                                    "cp_sha256": manifest["archive_sha256"]})
                manifest_meta = self.drive.upload_bytes(raw, folder, manifest["snapshot"] + ".json",
                                                        app_properties=summary_properties(manifest))
                report["imported"].append({"project": project, "snapshot": manifest["snapshot"],
                                           "archive_bytes": manifest["archive_bytes"],
                                           "files": len(manifest["files"]),
                                           "archive_file_id": archive_meta["id"],
                                           "manifest_file_id": manifest_meta["id"]})
        return report


def connect(store):
    """Confirm the VM can reach Drive with the pushed token and report its account."""
    return {"drive_connected": True, "drive_email": store.require_drive().about().get("user", {}).get("emailAddress")}


OPERATIONS = {"status", "prepare", "snapshot", "snapshots", "restore", "seal", "launch",
              "job_status", "import_legacy", "connect"}


def dispatch(request):
    operation = request.get("operation")
    try:
        token = request.pop("drive", None)
        if token:
            path = os.environ.get(drive_api.TOKEN_ENV, drive_api.DEFAULT_TOKEN_FILE)
            drive_api.write_token_file(path, token)
            os.environ[drive_api.TOKEN_ENV] = path
        store = Store(folder=request.pop("folder", DEFAULT_FOLDER),
                      snapshot_limit=request.pop("snapshot_limit", DEFAULT_SNAPSHOT_LIMIT))
        operation = request.pop("operation")
        if operation not in OPERATIONS:
            raise ValueError("Unsupported persistence operation.")
        if operation == "connect":
            result = connect(store)
        elif operation in {"snapshot", "restore"}:
            with store.project_lock(request["project"]):
                result = getattr(store, operation)(**request)
        else:
            result = getattr(store, operation)(**request)
    except Exception as error:
        # The local side reports this reason instead of only "no result".
        print(ERROR_PREFIX + json.dumps({"operation": operation, "error": str(error),
                                         "type": type(error).__name__}), flush=True)
        raise
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


def _worker_main(argv):
    project = valid_name(argv[argv.index("--worker") + 1])
    scratch = argv[argv.index("--scratch") + 1] if "--scratch" in argv else "/content/colab-persist"
    spec = json.loads((Path(scratch) / project / "job-spec.json").read_text())
    Store(scratch=spec.get("scratch", scratch), folder=spec["folder"],
          snapshot_limit=spec["snapshot_limit"]).worker(project)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main(sys.argv)
    else:
        dispatch(json.loads(sys.stdin.read()))
