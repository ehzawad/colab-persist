"""Publish caller-completed, multi-file checkpoints as one local directory.

This is a generic filesystem helper, not a Trainer callback. The caller chooses
the required files and must stop all writers before leaving the context. Their
presence cannot prove that all state needed to resume training was saved. A
successful publication says nothing about upload or Google Drive durability.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Iterator
import uuid

from .remote import excluded


METADATA_FILE = "checkpoint.json"
_RESERVED = {"schema", "generation", "required_files"}


def _required_paths(required_files: list[str]) -> list[str]:
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("required_files must be a nonempty list of relative file paths.")
    paths = []
    for name in required_files:
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("Required files must be nonempty relative POSIX paths.")
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("Required files cannot contain absolute paths or traversal.")
        if path.parts[0] == METADATA_FILE:
            raise ValueError(f"{METADATA_FILE} is reserved for publication metadata.")
        if excluded(Path(name)):
            raise ValueError("Required checkpoint files cannot match workspace snapshot exclusions.")
        if name in paths:
            raise ValueError("Required file paths must be unique.")
        paths.append(name)
    return paths


def _validate_tree(staging: Path) -> None:
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("Checkpoint staging must remain a real directory.")
    for directory, names, files in os.walk(staging, followlinks=False):
        for name in [*names, *files]:
            entry = Path(directory, name)
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Checkpoint paths cannot contain symbolic links.")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError("Checkpoint paths must be ordinary files or directories.")


@contextmanager
def atomic_checkpoint(root: Path, generation: str, required_files: list[str],
                      metadata: dict) -> Iterator[Path]:
    """Yield a staging directory and publish it only after local validation.

    ``root`` should be inside the managed workspace. Write every application
    state file into the yielded path; supply a nonempty list of required files.
    Each must be a nonempty regular file. Symbolic links and special files are
    rejected anywhere in the generation. ``metadata`` may include
    ``dataset_manifest_sha256`` to identify the exact dataset version; it is not
    required or independently verified by this helper.

    Cooperating publishers hold a root-level flock throughout the context.
    Existing generations are never intentionally replaced. An exception leaves
    the uniquely named ``.partial`` directory for diagnosis; workspace snapshots
    exclude it. No old checkpoints are deleted. Do not modify a generation after
    publication, or write concurrently from background processes.
    """
    if not isinstance(generation, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", generation):
        raise ValueError("Generation must be 1-64 letters, digits, underscores or hyphens.")
    required = _required_paths(required_files)
    if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
        raise ValueError("metadata must be a dictionary with string keys.")
    if _RESERVED.intersection(metadata):
        raise ValueError("metadata cannot replace schema, generation or required_files.")
    document = {"schema": 1, "generation": generation, "required_files": required, **metadata}
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"

    root = Path(root)
    if root.is_symlink():
        raise ValueError("Checkpoint root cannot be a symbolic link.")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Checkpoint root must be a real directory.")
    root = root.resolve()
    lock_path = root / ".publish-lock.partial"
    if lock_path.is_symlink():
        raise ValueError("Checkpoint lock cannot be a symbolic link.")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "a+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Checkpoint lock must be an ordinary file.")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            destination = root / generation
            if os.path.lexists(destination):
                raise FileExistsError(f"Checkpoint generation already exists: {generation}")
            staging = root / f"{generation}-{uuid.uuid4().hex}.partial"
            staging.mkdir(mode=0o700)
            yield staging
            _validate_tree(staging)
            for name in required:
                path = staging / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Required checkpoint file is missing or empty: {name}")
            if os.path.lexists(staging / METADATA_FILE):
                raise ValueError(f"{METADATA_FILE} is reserved for publication metadata.")
            with (staging / METADATA_FILE).open("x", encoding="utf-8") as target:
                target.write(serialized)
                target.flush()
                os.fsync(target.fileno())
            if os.path.lexists(destination):
                raise FileExistsError(f"Checkpoint generation already exists: {generation}")
            staging.rename(destination)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
