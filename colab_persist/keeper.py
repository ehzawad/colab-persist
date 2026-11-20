"""Keep a running job's Drive access fresh from this computer.

The VM only ever holds a Drive token that expires within an hour; this computer
renews it whenever it talks to the VM. This helper runs in the background on this
computer (never on the VM) while a managed job is active, and sends the same
short-lived, Drive-only token every command sends, so periodic checkpoints keep
saving while this computer is awake and online. It exits once no job is active,
so it never keeps an idle runtime busy, and it holds no credential of its own.
"""
import fcntl
import os
import sys
import time
from datetime import datetime, timezone

from . import backend

RENEW_SECONDS = 10 * 60       # Shorter than the renewal margin, so the VM's token never lapses.
RETRY_SECONDS = 60            # After a failure (offline, VM busy), try again soon.
TICK_SECONDS = 30             # Wall-clock checks catch up quickly after this computer sleeps.
GIVE_UP_SECONDS = 24 * 3600   # Matches how long a finished job waits for Drive.
ACTIVE_STATES = {"starting", "running", "saving", "awaiting_drive"}
LOG_LIMIT_BYTES = 1024 * 1024


def active_jobs(status):
    return sorted(name for name, item in (status.get("workspaces") or {}).items()
                  if item.get("active") or item.get("job") in ACTIVE_STATES)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def wanted_since(moment):
    """True when a command asked for a keeper after `moment` (a job started meanwhile)."""
    try:
        return (backend.CONFIG_DIR / backend.KEEPER_WANTED).stat().st_mtime >= moment
    except OSError:
        return False


def run(*, clock=time.time, sleep=time.sleep):
    """Renew until no managed job is active; return why it stopped."""
    email = backend.config().get("expected_email")
    due = clock() + RENEW_SECONDS  # The command that started this helper just sent a fresh token.
    failing_since = None
    while True:
        while clock() < due:
            sleep(min(TICK_SECONDS, max(0.0, due - clock())))
        checked_at = clock()
        try:
            if backend.config().get("expected_email") != email:
                log("The selected account changed; stopping.")
                return "account_changed"
            item = backend.runtime_info()
            if item is None:
                log("No managed runtime is running; stopping.")
                return "no_runtime"
            jobs = active_jobs(backend.remote_call("status", item=item, timeout=120))  # Sends a fresh token.
        except Exception as error:  # Offline, asleep or a busy VM: keep trying.
            failing_since = failing_since or checked_at
            if clock() - failing_since >= GIVE_UP_SECONDS:
                log(f"Could not reach the runtime for 24 hours; stopping. Last error: {error}")
                return "unreachable"
            log(f"Could not renew Drive access yet; retrying: {str(error).splitlines()[0][:300]}")
            due = clock() + RETRY_SECONDS
            continue
        failing_since = None
        if not jobs and not wanted_since(checked_at):
            log("No managed job is active; stopping.")
            return "idle"
        log("Renewed Drive access" + (f" for {', '.join(jobs)}." if jobs else "."))
        due = clock() + RENEW_SECONDS


def main():
    backend.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(backend.CONFIG_DIR / backend.KEEPER_LOCK, "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0  # Another keeper already renews for this computer.
    log_path = backend.CONFIG_DIR / backend.KEEPER_LOG
    if log_path.exists() and log_path.stat().st_size > LOG_LIMIT_BYTES:
        log_path.write_text("")
    log(f"Started (pid {os.getpid()}).")
    try:
        run()
    except Exception as error:
        log(f"Stopped after an unexpected error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
