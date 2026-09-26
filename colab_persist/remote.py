"""Dependency-free checkpoint engine, copied to the disposable Linux runtime."""
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
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".ssh", ".config",
            ".codex", ".claude", ".gemini", "node_modules", ".DS_Store"}
RESULT_PREFIX = "COLAB_PERSIST_RESULT="


def valid_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
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


def inventory(root: Path) -> dict:
    """Only regular files inside the workspace; never follow symbolic links."""
    result = {}
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
            sha = digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError(f"File changed during checkpoint: {relative}. Retry after its writer finishes.")
            result[relative.as_posix()] = {"size": before.st_size, "sha256": sha,
                                           "mode": stat.S_IMODE(before.st_mode)}
    return result


def extract_archive(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe path or special file in checkpoint archive.")
        source.extractall(target, members=members, filter="data")


class Store:
    def __init__(self, scratch="/content/colab-persist", mount="/content/drive", folder="Colab-CUDA"):
        self.scratch = Path(scratch)
        self.mount = Path(mount)
        self.root = self.mount / "MyDrive" / valid_name(folder) / "projects"

    def require_drive(self):
        if not os.path.ismount(self.mount) or not (self.mount / "MyDrive").is_dir():
            raise RuntimeError("Drive is not mounted. Run `colab-persist mount` before saving or running.")

    def workspace(self, project):
        return self.scratch / valid_name(project) / "workspace"

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

    def snapshots(self, project):
        self.require_drive()
        path = self.root / valid_name(project)
        result = []
        for manifest in sorted(path.glob("*.json"), reverse=True):
            item = json.loads(manifest.read_text())
            if item.get("schema") == 1 and item.get("snapshot") == manifest.stem:
                result.append(item)
        return result

    def snapshot(self, project, reason="manual"):
        self.require_drive()
        workspace = self.workspace(project)
        if not workspace.is_dir():
            raise RuntimeError(f"Workspace {project} has not been prepared.")
        files = inventory(workspace)
        previous = self.snapshots(project)
        if previous and previous[0]["files"] == files:
            archive = self.root / project / (previous[0]["snapshot"] + ".tar.gz")
            if digest(archive) == previous[0]["archive_sha256"]:
                return {**previous[0], "reused": True, "durability": "pending_drive_flush"}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        identifier = stamp + "-" + uuid.uuid4().hex[:8]
        destination = self.root / project
        destination.mkdir(parents=True, exist_ok=True)
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
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "reason": reason, "files": files, "archive_sha256": digest(archive),
                        "archive_bytes": archive.stat().st_size}
            partial = destination / (identifier + ".tar.gz.partial")
            shutil.copyfile(archive, partial)
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            if digest(partial) != manifest["archive_sha256"]:
                raise RuntimeError("Drive archive checksum mismatch. Previous snapshots are preserved.")
            partial.replace(destination / archive.name)
            # Manifest is the commit marker and is published only after the archive verifies.
            write_json(destination / (identifier + ".json"), manifest)
        return {**manifest, "reused": False, "durability": "pending_drive_flush"}

    def restore(self, project, snapshot="latest"):
        self.require_drive()
        workspace = self.workspace(project)
        if workspace.exists() and any(workspace.iterdir()):
            raise RuntimeError("Restore requires an empty workspace; existing work is never overwritten.")
        choices = self.snapshots(project)
        if snapshot == "latest":
            manifest = choices[0] if choices else None
        else:
            manifest = next((s for s in choices if s["snapshot"] == snapshot), None)
        if manifest is None:
            raise RuntimeError("No completed snapshot found.")
        archive = self.root / project / (manifest["snapshot"] + ".tar.gz")
        if digest(archive) != manifest["archive_sha256"]:
            raise RuntimeError("Checkpoint archive checksum mismatch; restore aborted.")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="restore-", dir=workspace.parent) as temporary:
            staging = Path(temporary) / "workspace"
            staging.mkdir()
            extract_archive(archive, staging)
            if inventory(staging) != manifest["files"]:
                raise RuntimeError("Restored files do not match the checkpoint manifest.")
            if workspace.exists():
                workspace.rmdir()  # Empty only; never removes user files.
            staging.rename(workspace)
        return {"project": project, "snapshot": manifest["snapshot"],
                "restored_files": len(manifest["files"]), "workspace": str(workspace),
                "archive_sha256": manifest["archive_sha256"]}

    def prepare(self, project):
        with self.project_lock(project):
            self.require_drive()
            workspace = self.workspace(project)
            restored = None
            if not workspace.exists() or not any(workspace.iterdir()):
                if self.snapshots(project):
                    restored = self.restore(project)
                else:
                    workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "outputs").mkdir(exist_ok=True)
            return {"workspace": str(workspace), "restored": restored}

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
                    jobs[base.name] = {"active": active, "workspace": str(base / "workspace")}
        return {"drive_mounted": os.path.ismount(self.mount), "workspaces": jobs,
                "drive_root": str(self.root)}

    def run(self, project, script, arguments, interval=60):
        self.require_drive()
        if not isinstance(interval, int) or interval < 10:
            raise ValueError("Checkpoint interval must be at least 10 seconds.")
        workspace = self.workspace(project)
        relative = PurePosixPath(script)
        if relative.is_absolute() or ".." in relative.parts or not script.endswith(".py"):
            raise ValueError("Script must be a relative Python file inside the workspace.")
        entry = workspace / script
        if entry.is_symlink() or not entry.resolve().is_relative_to(workspace.resolve()) or not entry.is_file():
            raise ValueError("Script is missing or is outside the workspace.")
        with self.project_lock(project):
            log = workspace / "outputs" / "run.log"
            env = os.environ.copy()
            env["COLAB_WORKSPACE"] = str(workspace)
            env["COLAB_OUTPUT_DIR"] = str(workspace / "outputs")
            failures = []
            with log.open("ab", buffering=0) as output:
                process = subprocess.Popen([sys.executable, "-u", str(entry), *arguments],
                                           cwd=workspace, env=env, stdout=output, stderr=output)
                last_save = time.monotonic()
                try:
                    while process.poll() is None:
                        time.sleep(1)
                        if time.monotonic() - last_save >= interval:
                            try:
                                self.snapshot(project, "periodic")
                            except Exception as error:
                                failures.append(str(error))
                            last_save = time.monotonic()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            write_json(workspace / "outputs" / "last-run.json",
                       {"script": script, "arguments": arguments, "exit_code": process.returncode,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "periodic_checkpoint_errors": failures[-10:]})
            saved = self.snapshot(project, "script_exit")
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8000))
                tail = stream.read().decode(errors="replace")
            return {"exit_code": process.returncode, "checkpoint": saved,
                    "log_tail": tail, "periodic_checkpoint_errors": failures[-10:]}

    def seal(self):
        """Checkpoint every managed workspace, then push Drive writes before allowing shutdown."""
        projects = list(self.status()["workspaces"])
        with contextlib.ExitStack() as stack:
            for project in projects:
                stack.enter_context(self.project_lock(project))
            receipt_path = self.scratch / "flush-receipt.json"
            if not os.path.ismount(self.mount) and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                saved = {item["project"]: item["files"] for item in receipt["checkpoints"]}
                if set(projects) == set(saved) and all(inventory(self.workspace(p)) == saved[p] for p in projects):
                    return receipt
                raise RuntimeError("Workspace changed after Drive was flushed. Remount Drive before stopping.")
            self.require_drive()
            checkpoints = [self.snapshot(project, "safe_stop") for project in projects]
            from google.colab import drive
            drive.flush_and_unmount(timeout_ms=300_000)
            if os.path.ismount(self.mount):
                raise RuntimeError("Drive is still mounted after flush; shutdown refused.")
            if any(inventory(self.workspace(item["project"])) != item["files"] for item in checkpoints):
                raise RuntimeError("Workspace changed while Drive was flushing. Remount and save again; shutdown refused.")
            receipt = {"drive_flushed": True, "checkpoints": checkpoints,
                       "flushed_at": datetime.now(timezone.utc).isoformat()}
            for item in receipt["checkpoints"]:
                item["durability"] = "drive_flush_confirmed"
            self.scratch.mkdir(parents=True, exist_ok=True)
            write_json(receipt_path, receipt)
            return receipt


def dispatch(request):
    store = Store(folder=request.pop("folder", "Colab-CUDA"))
    operation = request.pop("operation")
    if operation not in {"status", "prepare", "snapshot", "snapshots", "restore", "run", "seal"}:
        raise ValueError("Unsupported persistence operation.")
    if operation in {"snapshot", "restore"}:
        with store.project_lock(request["project"]):
            result = getattr(store, operation)(**request)
    else:
        result = getattr(store, operation)(**request)
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    dispatch(json.loads(sys.stdin.read()))
