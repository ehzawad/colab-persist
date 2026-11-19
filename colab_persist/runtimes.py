"""Private ledger of the runtimes this tool created: one generation per session name.

Each new VM gets a fresh runtime ID, recorded before the create request, so an
interrupted ``start`` resumes the same creation instead of allocating twice, and
``stop`` deletes only the exact generation it saved. Browser and other runtimes
are listed but never adopted or deleted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import uuid

from . import accounts

ABANDONED = "_abandoned"  # Not a valid session name, so it never collides with one.


def ledger_path(config_dir: Path, email: str) -> Path:
    return accounts.account_dir(config_dir, email) / "runtimes.json"


def load(config_dir, email) -> dict:
    try:
        return json.loads(ledger_path(config_dir, email).read_text())
    except FileNotFoundError:
        return {}


def _save(config_dir, email, data):
    path = ledger_path(config_dir, email)
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(data, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def new_runtime_id(session: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", session.lower()).strip("-")[:40] or "session"
    if not slug[0].isalpha():
        slug = "s" + slug
    return f"cp-{slug}-{secrets.token_hex(3)}"


def entry(config_dir, email, session):
    return load(config_dir, email).get(session)


def begin(config_dir, email, session, spec) -> dict:
    """Return the pending creation for this session and spec, or record a new one."""
    data = load(config_dir, email)
    current = data.get(session)
    if current and current.get("state") == "pending":
        if current.get("spec") == spec:
            return current
        raise RuntimeError("A different runtime creation is still pending for this session. Run "
                           "`colab-persist start` with its original hardware first, or `colab-persist sessions`.")
    if current and current.get("state") == "ready":
        raise RuntimeError("A managed runtime is already recorded for this session. Run `colab-persist status`.")
    record = {"runtime_id": new_runtime_id(session), "request_id": str(uuid.uuid4()), "spec": spec,
              "state": "pending", "created_at": datetime.now(timezone.utc).isoformat()}
    data[session] = record
    _save(config_dir, email, data)
    return record


def mark_ready(config_dir, email, session, runtime_id, endpoint):
    data = load(config_dir, email)
    record = data.get(session)
    if not record or record["runtime_id"] != runtime_id:
        raise RuntimeError("The runtime ledger changed during creation; refusing to adopt another runtime.")
    record.update(state="ready", endpoint=endpoint)
    _save(config_dir, email, data)
    return record


def forget(config_dir, email, session, runtime_id):
    """Remove one exact generation; never a replacement recorded later."""
    data = load(config_dir, email)
    record = data.get(session)
    if record and record.get("runtime_id") == runtime_id:
        del data[session]
        _save(config_dir, email, data)


def abandon(config_dir, email, session, runtime_id):
    """Give up waiting for a creation, but remember its exact ID in case it still appears."""
    data = load(config_dir, email)
    record = data.get(session)
    if record and record.get("runtime_id") == runtime_id:
        data.setdefault(ABANDONED, {})[runtime_id] = {
            **record, "session": session, "abandoned_at": datetime.now(timezone.utc).isoformat()}
        del data[session]
        _save(config_dir, email, data)


def abandoned(config_dir, email) -> dict:
    return load(config_dir, email).get(ABANDONED, {})


def adopt(config_dir, email, session, runtime_id, endpoint):
    """Record a runtime this tool created earlier (matched by its exact ID) as the managed one."""
    data = load(config_dir, email)
    current = data.get(session)
    if current and current.get("state") == "ready":
        raise RuntimeError("A managed runtime is already recorded for this session.")
    record = data.get(ABANDONED, {}).pop(runtime_id, None)
    if record is None:
        raise RuntimeError("Only runtimes this tool created can be adopted.")
    record = {k: v for k, v in record.items() if k not in ("session", "abandoned_at")}
    record.update(state="ready", endpoint=endpoint)
    data[session] = record
    _save(config_dir, email, data)
    return record


def prune_abandoned(config_dir, email, present_ids, max_age_hours=48):
    """Forget abandoned IDs that never appeared within two days."""
    data = load(config_dir, email)
    stale = []
    for runtime_id, record in data.get(ABANDONED, {}).items():
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(record.get("abandoned_at", ""))
        except ValueError:
            continue
        if runtime_id not in present_ids and age.total_seconds() > max_age_hours * 3600:
            stale.append(runtime_id)
    if stale:
        for runtime_id in stale:
            del data[ABANDONED][runtime_id]
        _save(config_dir, email, data)
