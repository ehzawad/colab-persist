"""Account-isolated Colab session mappings, with verified legacy adoption.

The official CLI's default session file has no account namespace. Synchronizing
that file under another Google account can prune a still-running VM's mapping.
Keep our mappings separate and consult the old file only to recover a verified
assignment during an upgrade.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from google.auth.transport.requests import AuthorizedSession
from colab_cli.auth import AuthProvider
from colab_cli.client import Client, Prod
from colab_cli.common import State
from colab_cli.state import StateStore


def session_path(config_dir: Path, email: str) -> Path:
    """Return an account's private state path without placing its email in it."""
    normalized = email.strip().lower()
    if not normalized:
        raise ValueError("An account email is required for session isolation.")
    account_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    accounts = Path(config_dir) / "accounts"
    accounts.mkdir(parents=True, exist_ok=True, mode=0o700)
    accounts.chmod(0o700)
    directory = accounts / account_id
    directory.mkdir(exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    destination = directory / "sessions.json"
    if destination.exists():
        destination.chmod(0o600)
    return destination


def _legacy_session(name: str):
    """Read one old mapping; never synchronize or modify the shared store."""
    path = Path.home() / ".config" / "colab-cli" / "sessions.json"
    if not path.is_file():
        return None
    return StateStore(str(path)).get(name)


def synced_state(config_dir: Path, cfg: dict, credentials):
    """Return (sessions, assignments) using already-verified Google credentials.

    ``credentials`` is a google.auth Credentials object. Server errors propagate
    before any mappings are adopted, refreshed, or pruned; an authentication
    failure must never be reported as an empty account.
    """
    path = session_path(config_dir, cfg["expected_email"])
    state = State()
    state.config_path = str(path)
    state.auth_provider = AuthProvider.ADC
    authorized = AuthorizedSession(credentials)
    state._client = Client(Prod(), authorized)
    try:
        # Avoid State.sync_sessions(): its empty-store branch catches SystemExit
        # from authentication and returns an apparently empty assignment list.
        assignments = state.client.list_assignments()
    finally:
        authorized.close()
    by_endpoint = {assignment.endpoint: assignment for assignment in assignments}
    sessions = state.store.list()

    # Reconcile only our account-owned file. Using prune_session here would also
    # write to the upstream CLI's shared history directory.
    for name, item in list(sessions.items()):
        if item.endpoint not in by_endpoint:
            state.store.remove(name)
            del sessions[name]

    name = cfg.get("session", "cuda")
    if name not in sessions:
        legacy = _legacy_session(name)
        if legacy is not None and legacy.endpoint in by_endpoint:
            sessions[name] = legacy

    for item in sessions.values():
        proxy = by_endpoint[item.endpoint].runtime_proxy_info
        item.token = proxy.token
        item.url = proxy.url
        item.token_expires_at = proxy.expires_at()
        state.store.add(item)
    if path.exists():
        path.chmod(0o600)
    return sessions, assignments
