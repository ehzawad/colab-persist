"""Local control plane. Credentials stay on this computer; the VM receives project
files and short-lived Drive tokens only."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid

from . import accounts, colab_api, datasets, drive_api, remote, runtimes

CONFIG_DIR = Path.home() / ".config" / "colab-persist"
ENGINE_ROOT = remote.ENGINE_ROOT
ENGINE_COMMAND = f"cd {ENGINE_ROOT} && python3 -m colab_persist.remote"
ENGINE_FILES = ("__init__.py", "remote.py", "drive_api.py", "datasets.py", "checkpoints.py")
SSH_CONTROL_DIR = Path.home() / ".ssh"
SSH_BUSY = ("Already-active SSH session", "HTTP 429")
LEGACY_CLI = "google-colab-cli==0.7.4"
POLL_SECONDS = 10
TOKEN_MARGIN_SECONDS = 15 * 60
TERMINAL_JOB_STATES = {"finished", "exited_unsaved", "refused", "none", "lost"}
_TOKENS: dict = {}


# -- configuration ---------------------------------------------------------------------
def config():
    path = CONFIG_DIR / "config.json"
    if not path.exists():
        raise RuntimeError("Configure your account first: colab-persist configure --email YOUR_GOOGLE_EMAIL")
    value = json.loads(path.read_text())
    if not value.get("expected_email"):
        raise RuntimeError("The configuration must specify expected_email.")
    accounts.normalize_email(value["expected_email"])
    if value.get("storage") != 2:
        # v0.3 wrote through Colab's own Drive mount. drive.file cannot see that folder, so the
        # API keeps checkpoints in a separate folder and import-legacy copies old ones over.
        value["legacy_drive_folder"] = value.get("drive_folder", remote.LEGACY_FOLDER)
        value["drive_folder"] = remote.DEFAULT_FOLDER
        value["storage"] = 2
    value.setdefault("drive_folder", remote.DEFAULT_FOLDER)
    value.setdefault("legacy_drive_folder", remote.LEGACY_FOLDER)
    remote.valid_name(value.get("session", "cuda"))
    remote.valid_name(value["drive_folder"])
    remote.valid_name(value["legacy_drive_folder"])
    limit = value.get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT)
    if type(limit) is not int or limit <= 0:
        raise ValueError("snapshot_limit_bytes must be a positive integer.")
    meta = accounts.meta(CONFIG_DIR, value["expected_email"])
    value["scopes"] = list(accounts.login_scopes(bool(meta.get("full_drive"))))
    return value


def write_config(value):
    stored = {k: v for k, v in value.items() if k != "scopes"}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    remote.write_json(CONFIG_DIR / "config.json", stored)
    (CONFIG_DIR / "config.json").chmod(0o600)


@contextlib.contextmanager
def operation_lock():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (CONFIG_DIR / "operation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another persistence operation is active. Wait for it to finish.")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextlib.contextmanager
def waiting_lock(seconds=300):
    """Like operation_lock, but waits for a short concurrent command (such as status) to finish."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = time.monotonic() + seconds
    with (CONFIG_DIR / "operation.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Another persistence operation is active. Wait for it to finish.") from None
                time.sleep(2)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def quota_project(cfg):
    project = accounts.meta(CONFIG_DIR, cfg["expected_email"]).get("quota_project")
    if not project:
        raise RuntimeError("No Cloud project is set for this account. Run `colab-persist setup --project "
                           "YOUR_PROJECT_ID` once (free; enables the Colab and Drive APIs, no billing).")
    return project


def full_drive(cfg):
    return bool(accounts.meta(CONFIG_DIR, cfg["expected_email"]).get("full_drive"))


def api_for(cfg, credentials=None):
    return colab_api.ColabAPI(credentials or accounts.verified_credentials(cfg), quota_project(cfg))


def drive_token(cfg):
    """A cached short-lived Drive token; refreshed when less than 15 minutes remain."""
    path = accounts.credentials_path(cfg)
    stamp = path.stat().st_mtime_ns if path and path.exists() else None
    # A long-running MCP server must notice a reauth, a new project or a scope change made elsewhere.
    key = (cfg["expected_email"], cfg["drive_folder"], quota_project(cfg), full_drive(cfg), str(path), stamp)
    token = _TOKENS.get(key)
    if not token or token["expires_at"] - time.time() < TOKEN_MARGIN_SECONDS:
        token = accounts.drive_token(cfg, quota_project(cfg), cfg["drive_folder"], full_drive=full_drive(cfg))
        # Every token that reaches the VM is first proven to belong to the selected account.
        check_drive_identity(cfg, drive_api.DriveClient(drive_api.TokenSource(token["token"],
                                                                               token["quota_project"])))
        _TOKENS[key] = token
    return dict(token)


def local_drive(cfg):
    token = drive_token(cfg)
    return drive_api.DriveClient(drive_api.TokenSource(token["token"], token["quota_project"]))


def check_drive_identity(cfg, drive=None):
    about = (drive or local_drive(cfg)).about()
    email = about.get("user", {}).get("emailAddress", "").lower()
    if email != cfg["expected_email"].lower():
        raise RuntimeError(f"Drive account mismatch: expected {cfg['expected_email']}, got {email or 'none'}. "
                           "Nothing was written.")
    return about


def identity():
    cfg = config()
    accounts.verified_credentials(cfg)
    return cfg["expected_email"].lower()


# -- accounts --------------------------------------------------------------------------
def _service_disabled(error):
    reason = getattr(error, "reason", None)
    return reason in ("SERVICE_DISABLED", "accessNotConfigured") or "SERVICE_DISABLED" in str(error)


def _after_enabling(api, project, check, attempts=12):
    """Run a check, enabling the two free APIs and waiting for them to take effect if needed."""
    enabled = False
    for attempt in range(attempts):
        try:
            return check()
        except (colab_api.ColabAPIError, drive_api.DriveError) as error:
            if not _service_disabled(error) or attempt == attempts - 1:
                raise
            if not enabled:
                api.enable_services(project)  # First use of this project.
                enabled = True
            time.sleep(5)  # Newly enabled APIs can take a minute to take effect.


def verify_account(cfg, project):
    credentials = accounts.verified_credentials(cfg)
    result = {"account": cfg["expected_email"], "account_verified": True}
    if not project:
        return {**result, "drive_ready": False, "next": "colab-persist setup --project YOUR_PROJECT_ID"}
    api = colab_api.ColabAPI(credentials, project)
    result["active_runtimes"] = len(_after_enabling(api, project, api.list_runtimes))
    full = cfg.get("_full_drive", False)
    token = accounts.drive_token(cfg, project, cfg["drive_folder"], full_drive=full)
    drive = drive_api.DriveClient(drive_api.TokenSource(token["token"], project))
    _after_enabling(api, project, lambda: check_drive_identity(cfg, drive))
    return {**result, "drive_ready": True}


def login(email, *, reauth=False, no_launch_browser=False, no_switch=False, project=None,
          client_id_file=None):
    """Verify and save private credentials (Colab + Drive), optionally selecting that account."""
    email = accounts.normalize_email(email)
    with operation_lock():
        previous = config()
        switching = not no_switch and email != previous["expected_email"].lower()

        def require_old_account_stopped():
            if not switching:
                return
            try:
                active = api_for(previous).list_runtimes()
            except Exception as error:
                raise RuntimeError("Could not verify that the current account has no running Colab runtimes; "
                                   "switching refused. " + str(error)) from None
            if active:
                raise RuntimeError("The current Google account still has active Colab runtimes. "
                                   "Save and stop them before switching accounts.")
            pending = [s for s, r in runtimes.load(CONFIG_DIR, previous["expected_email"]).items()
                       if s != runtimes.ABANDONED and r.get("state") == "pending"]
            if pending:
                raise RuntimeError("The current account has a runtime creation still pending. Run "
                                   "`colab-persist start` (or `status`) on it first, then switch.")

        require_old_account_stopped()
        directory = accounts.account_dir(CONFIG_DIR, email)
        saved = directory / "credentials.json"
        full = bool(client_id_file)
        account_meta = accounts.meta(CONFIG_DIR, email)
        project = project or account_meta.get("quota_project")
        candidate = {**previous, "expected_email": email, "credentials_file": str(saved),
                     "scopes": list(accounts.login_scopes(full)), "_full_drive": full}
        if reauth or not saved.exists() or full != bool(account_meta.get("full_drive")):
            # Isolate gcloud's ADC output and configuration; the user's normal ADC
            # and gcloud account selection are never overwritten.
            with tempfile.TemporaryDirectory(prefix="login-", dir=directory) as temporary:
                env = os.environ.copy()
                env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
                env["CLOUDSDK_CONFIG"] = temporary
                command = ["gcloud", "auth", "application-default", "login", email,
                           "--disable-quota-project", "--scopes=" + ",".join(accounts.login_scopes(full))]
                if client_id_file:
                    command.append("--client-id-file=" + str(Path(client_id_file).expanduser().absolute()))
                if no_launch_browser:
                    command.append("--no-launch-browser")
                result = subprocess.run(command, env=env)
                if result.returncode:
                    raise RuntimeError("Google login did not finish; account selection is unchanged.")
                staged = Path(temporary) / "application_default_credentials.json"
                candidate["credentials_file"] = str(staged)
                verification = verify_account(candidate, project)  # Before committing anything.
                require_old_account_stopped()  # A runtime may have started during browser consent.
                staged.chmod(0o600)
                staged.replace(saved)
        else:
            verification = verify_account(candidate, project)
            require_old_account_stopped()
        candidate["credentials_file"] = str(saved)
        candidate.pop("_full_drive", None)
        accounts.save_meta(CONFIG_DIR, email, full_drive=full, **({"quota_project": project} if project else {}))
        _TOKENS.clear()
        if not no_switch:
            write_config(candidate)
        return {**verification, "switched": switching,
                "selected_account": previous["expected_email"] if no_switch else email,
                "saved_only": no_switch, "drive_scope": "drive" if full else "drive.file",
                "global_gcloud_credentials_changed": False,
                "next": verification.get("next") or ("colab-persist login --email " + email if no_switch
                                                     else "colab-persist status")}


def setup(project):
    """Enable the two free APIs on the user's Cloud project and remember it for this account."""
    if not isinstance(project, str) or not project.strip():
        raise ValueError("Provide your Google Cloud project ID.")
    project = project.strip()
    with operation_lock():
        cfg = config()
        credentials = accounts.verified_credentials(cfg)
        api = colab_api.ColabAPI(credentials, project)
        services = api.enable_services(project)
        accounts.save_meta(CONFIG_DIR, cfg["expected_email"], quota_project=project)
        _TOKENS.clear()
        tier = _after_enabling(api, project, api.subscription)
        result = {"account": cfg["expected_email"], "project_saved": True, "services": services,
                  "colab_tier": tier.replace("SUBSCRIPTION_TIER_", "").lower()}
        try:
            _after_enabling(api, project, lambda: check_drive_identity(cfg))
            result["drive_ready"] = True
        except RuntimeError as error:
            result.update(drive_ready=False, drive_error=str(error))
        write_config(cfg)
        return result


# -- runtimes --------------------------------------------------------------------------
def managed_runtime(cfg, api):
    """(managed runtime or None, every runtime on the account, pending creation record)."""
    email, session = cfg["expected_email"], cfg.get("session", "cuda")
    everything = api.list_runtimes()
    by_id = {colab_api.runtime_id_of(r): r for r in everything}
    ledger = runtimes.load(CONFIG_DIR, email)
    record = ledger.get(session)
    if record and record["runtime_id"] not in by_id and record.get("state") == "ready":
        # A listing can miss a runtime another command just created; ask for it by exact ID.
        confirmed = api.get_runtime(record["runtime_id"])
        if confirmed is not None:
            by_id[record["runtime_id"]] = confirmed
            everything = [*everything, confirmed]
    if record and record["runtime_id"] in by_id:
        runtime = by_id[record["runtime_id"]]
        if not runtime.get("connectionInfo"):
            return None, everything, record if record.get("state") == "pending" else None
        if record.get("state") == "pending":
            runtimes.mark_ready(CONFIG_DIR, email, session, record["runtime_id"],
                                colab_api.endpoint_from_url(runtime["connectionInfo"]["url"]))
        return runtime, everything, None
    if record and record.get("state") == "ready":
        # Deleted elsewhere or reclaimed by Google (confirmed by ID); forget only our own entry.
        runtimes.forget(CONFIG_DIR, email, session, record["runtime_id"])
        record = None
    if record is None:
        # A creation given up on earlier may still have produced a VM. It is ours by its exact ID.
        candidates = sorted(((rid, old) for rid, old in ledger.get(runtimes.ABANDONED, {}).items()
                             if rid in by_id and old.get("session") == session and
                             by_id[rid].get("connectionInfo")),
                            key=lambda item: item[1].get("created_at", ""), reverse=True)
        if candidates:
            runtime_id = candidates[0][0]
            runtimes.adopt(CONFIG_DIR, email, session, runtime_id,
                           colab_api.endpoint_from_url(by_id[runtime_id]["connectionInfo"]["url"]))
            return by_id[runtime_id], everything, None
    runtimes.prune_abandoned(CONFIG_DIR, email, set(by_id))
    return None, everything, record


def runtime_info():
    cfg = config()
    runtime, _, _ = managed_runtime(cfg, api_for(cfg))
    return runtime


def require_runtime():
    item = runtime_info()
    if item is None:
        raise RuntimeError("No managed runtime is active. Run `colab-persist start`.")
    return item


def specs():
    cfg = config()
    api = api_for(cfg)
    items = api.runtime_specs()
    return {"account": cfg["expected_email"], "tier": api.subscription().replace("SUBSCRIPTION_TIER_", "").lower(),
            "eligible": sorted(colab_api.spec_label(s["key"]) for s in items if s.get("eligible")),
            "not_eligible": sorted(colab_api.spec_label(s["key"]) for s in items if not s.get("eligible")),
            "default_gpu": cfg.get("gpu", "L4")}


def list_sessions():
    with operation_lock():
        cfg = config()
        api = api_for(cfg)
        runtime, everything, pending = managed_runtime(cfg, api)
        managed_id = colab_api.runtime_id_of(runtime) if runtime else None
        abandoned = runtimes.abandoned(CONFIG_DIR, cfg["expected_email"])
        result = {"account": cfg["expected_email"], "account_verified": True,
                  "active_runtimes": len(everything), "pending_creation": bool(pending),
                  "sessions": [{"runtime": colab_api.runtime_id_of(item),
                                "spec": colab_api.spec_label(item.get("runtimeSpec", {})),
                                "managed": colab_api.runtime_id_of(item) == managed_id,
                                "name": cfg.get("session", "cuda") if colab_api.runtime_id_of(item) == managed_id else None,
                                **({"abandoned_creation": True} if colab_api.runtime_id_of(item) in abandoned else {})}
                               for item in everything]}
        if any(item.get("abandoned_creation") for item in result["sessions"]):
            result["warning"] = ("A runtime colab-persist started earlier is running unmanaged and using compute "
                                 "units. Delete it in Colab (Runtime > Manage sessions), or stop the managed "
                                 "runtime so the next command adopts it.")
        return result


def checked(arguments, *, data=None, timeout=180, env=None):
    result = subprocess.run(arguments, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, env=env)
    if result.returncode:
        detail = (result.stderr + result.stdout).decode(errors="replace")[-4000:]
        if any(marker in detail for marker in SSH_BUSY):
            raise RuntimeError(
                "Colab accepts one SSH session per runtime and another client holds it: an interactive "
                "SSH, a native `colab ssh`, or another tool's open connection. Close it and retry; a "
                "multiplexed `ssh colab-cuda` connection closes with `ssh -O exit colab-cuda`.\n" + detail[-500:])
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")
    return result.stdout


def control_path(item):
    """One SSH connection per runtime, shared by colab-persist and the `colab-cuda` alias. Colab
    accepts a single SSH session per runtime, so separate connections would refuse each other
    (HTTP 429). The name is hashed to stay within macOS's 104-byte socket path limit."""
    digest = hashlib.sha256(colab_api.runtime_id_of(item).encode()).hexdigest()[:16]
    return SSH_CONTROL_DIR / ("cm-cp-" + digest)


def proxy_command(item):
    return shlex.join([sys.executable, "-m", "colab_persist.proxy", "--runtime", colab_api.runtime_id_of(item)])


def ssh_command(item):
    cfg = config()
    key = str(Path(cfg.get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser())
    return ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i", key,
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
            "-o", "UserKnownHostsFile=" + str(Path.home() / ".ssh/known_hosts_colab_persist"),
            "-o", "HostKeyAlias=colab-" + colab_api.runtime_id_of(item),
            # A backgrounded master (ControlPersist) lets this command return even while an
            # interactive session multiplexes on the same connection; it lingers 30 s after the last.
            "-o", "ControlMaster=auto", "-o", "ControlPath=" + str(control_path(item)),
            "-o", "ControlPersist=30",
            "-o", "ProxyCommand=" + proxy_command(item), "root@colab-runtime"]


def close_ssh(item):
    subprocess.run(["ssh", "-o", "ControlPath=" + str(control_path(item)), "-O", "exit", "root@colab-runtime"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def remote_call(operation, *, item=None, timeout=900, **parameters):
    cfg = config()
    item = item or require_runtime()
    request = {"operation": operation, "folder": cfg["drive_folder"],
               "snapshot_limit": cfg.get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT),
               "drive": drive_token(cfg), **parameters}
    payload = json.dumps(request).encode()
    try:
        raw = checked([*ssh_command(item), ENGINE_COMMAND], data=payload, timeout=timeout).decode(errors="replace")
    except RuntimeError as error:
        detail = str(error)
        if remote.ERROR_PREFIX in detail:
            return parse_result(detail, operation)  # Raises with the engine's own reason.
        missing = "No module named colab_persist" in detail or (
            ENGINE_ROOT in detail and ("can't cd" in detail or "No such file or directory" in detail))
        if not missing:
            raise
        install_engine(item)  # A runtime adopted after an interrupted start has no engine yet.
        raw = checked([*ssh_command(item), ENGINE_COMMAND], data=payload, timeout=timeout).decode(errors="replace")
    return parse_result(raw, operation)


@contextlib.contextmanager
def token_refresher(item, every_seconds=1800):
    """Keep the VM's Drive token fresh during one long remote operation."""
    stop = threading.Event()

    def refresh():
        while not stop.wait(every_seconds):
            try:
                remote_call("connect", item=item, timeout=120)
            except Exception:
                pass  # The operation itself reports any lasting Drive problem.
    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()


def parse_result(raw, operation=None):
    results = [line[len(remote.RESULT_PREFIX):] for line in raw.splitlines()
               if line.startswith(remote.RESULT_PREFIX)]
    if not results:
        errors = [line.split(remote.ERROR_PREFIX, 1)[1] for line in raw.splitlines()
                  if remote.ERROR_PREFIX in line]
        reason = ""
        if errors:
            with contextlib.suppress(ValueError, KeyError, TypeError):
                reason = ": " + json.loads(errors[-1])["error"]
        name = f"Remote operation {operation!r}" if operation else "The remote operation"
        stopped = " Runtime was NOT stopped." if operation == "seal" else ""
        raise RuntimeError(f"{name} returned no confirmed result{reason}.{stopped}\n" + raw[-2000:])
    return json.loads(results[-1])


def install_engine(item):
    cfg = config()
    source = io.BytesIO()
    package = Path(remote.__file__).parent
    with tarfile.open(fileobj=source, mode="w") as archive:
        for name in ENGINE_FILES:
            archive.add(package / name, arcname="colab_persist/" + name)
    for attempt in range(6):
        try:
            checked([*ssh_command(item), f"mkdir -p {ENGINE_ROOT} && tar -xf - -C {ENGINE_ROOT}"],
                    data=source.getvalue())
            break
        except RuntimeError:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))  # A fresh VM's SSH service can take a few seconds.
    folder = cfg["drive_folder"]
    # SSH shells do not inherit the notebook's CUDA library path.
    setup = f'''set -e
cat > /etc/profile.d/colab-cuda.sh <<'CUDA'
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64-nvidia:/usr/local/cuda/lib64${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export PYTHONPATH="{ENGINE_ROOT}${{PYTHONPATH:+:$PYTHONPATH}}"
export HF_HOME="${{HF_HOME:-/content/colab-persist-cache/huggingface}}"
export {drive_api.TOKEN_ENV}="{drive_api.DEFAULT_TOKEN_FILE}"
export {drive_api.FOLDER_ENV}="{folder}"
export COLAB_DATASET_ROOT="${{COLAB_DATASET_ROOT:-gdrive://{folder}/datasets}}"
CUDA
python3 - <<'PY'
from pathlib import Path
import site
p = Path.home() / '.bashrc'
line = 'source /etc/profile.d/colab-cuda.sh\\n'
old = p.read_text() if p.exists() else ''
if line not in old:
    p.write_text(line + old)
# Non-interactive SSH commands skip profile scripts; a .pth file still finds the helpers.
Path(site.getsitepackages()[0], 'colab_persist_engine.pth').write_text('{ENGINE_ROOT}\\n')
PY
'''.encode()
    checked([*ssh_command(item), "bash -s"], data=setup)


def write_ssh_alias(item):
    destination = config().get("ssh_alias_config")
    if not destination:
        return
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = Path(config().get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser()
    target.write_text(f'''# Managed by colab-persist start. Runtime-specific host key pinning.
Host colab-cuda
  HostName colab-runtime
  HostKeyAlias colab-{colab_api.runtime_id_of(item)}
  User root
  IdentityFile "{key}"
  IdentitiesOnly yes
  ProxyCommand {proxy_command(item)}
  StrictHostKeyChecking accept-new
  UserKnownHostsFile ~/.ssh/known_hosts_colab_persist
  ServerAliveInterval 30
  ServerAliveCountMax 3
  ForwardAgent no
  ControlMaster auto
  ControlPath "{control_path(item)}"
  ControlPersist 5m
''')
    target.chmod(0o600)


def start_runtime(gpu=None, tpu=None, cpu=False, high_mem=False):
    with operation_lock():
        cfg = config()
        credentials = accounts.verified_credentials(cfg)
        api = api_for(cfg, credentials)
        chosen_gpu = gpu or (None if (tpu or cpu) else cfg.get("gpu", "L4"))
        want = colab_api.wanted_spec(chosen_gpu, tpu, cpu, high_mem)  # Refuses conflicting choices.
        runtime, _, pending = managed_runtime(cfg, api)
        created = False
        if runtime is None and pending is not None:
            # Finish an interrupted creation first, so it can never run on unrecorded.
            try:
                runtime = api.create_runtime(pending["runtime_id"], pending["request_id"], pending["spec"],
                                             resume=True)
            except colab_api.ColabAPIError as error:
                if error.code == "PENDING":
                    raise
                # Remember the ID: if that VM still appears, it is adopted rather than left running.
                runtimes.abandon(CONFIG_DIR, cfg["expected_email"], cfg.get("session", "cuda"),
                                 pending["runtime_id"])
                raise
            else:
                runtimes.mark_ready(CONFIG_DIR, cfg["expected_email"], cfg.get("session", "cuda"),
                                    colab_api.runtime_id_of(runtime),
                                    colab_api.endpoint_from_url(runtime["connectionInfo"]["url"]))
                created = True
        if runtime is not None:
            if runtime.get("runtimeSpec") != want:
                raise RuntimeError(f"Runtime already uses {colab_api.spec_label(runtime.get('runtimeSpec', {}))}. "
                                   f"Save and stop it before selecting {colab_api.spec_label(want)}.")
        else:
            key = colab_api.choose_spec(api.runtime_specs(), want)
            check_drive_identity(cfg)  # Confirm Drive before spending compute units.
            record = runtimes.begin(CONFIG_DIR, cfg["expected_email"], cfg.get("session", "cuda"), key)
            try:
                runtime = api.create_runtime(record["runtime_id"], record["request_id"], key)
            except colab_api.ColabAPIError as error:
                if error.code != "PENDING":
                    # Remember the ID: if that VM still appears, it is adopted rather than left running.
                    runtimes.abandon(CONFIG_DIR, cfg["expected_email"], cfg.get("session", "cuda"),
                                     record["runtime_id"])
                raise
            runtimes.mark_ready(CONFIG_DIR, cfg["expected_email"], cfg.get("session", "cuda"),
                                colab_api.runtime_id_of(runtime),
                                colab_api.endpoint_from_url(runtime["connectionInfo"]["url"]))
            created = True
        install_engine(runtime)
        write_ssh_alias(runtime)
        return {"session": cfg.get("session", "cuda"), "runtime": colab_api.runtime_id_of(runtime),
                "gpu": colab_api.spec_label(runtime.get("runtimeSpec", {})), "created": created,
                **remote_call("status", item=runtime)}


def status():
    with operation_lock():
        cfg = config()
        credentials = accounts.verified_credentials(cfg)
        api = api_for(cfg, credentials)
        runtime, everything, pending = managed_runtime(cfg, api)
        summary = {"account": cfg["expected_email"], "account_verified": True,
                   "active_runtimes": len(everything), "drive_folder": f"My Drive/{cfg['drive_folder']}"}
        if runtime is None:
            return {**summary, "active": False, "pending_creation": bool(pending),
                    "next": "colab-persist start" if not everything or pending else "colab-persist sessions"}
        return {**summary, "active": True, "session": cfg.get("session", "cuda"),
                "runtime": colab_api.runtime_id_of(runtime),
                "gpu": colab_api.spec_label(runtime.get("runtimeSpec", {})),
                **remote_call("status", item=runtime)}


def mount_drive():
    """Connect Drive from this computer: no browser and no VM mount needed."""
    with operation_lock():
        cfg = config()
        drive = local_drive(cfg)
        about = check_drive_identity(cfg, drive)
        result = {"drive_connected": True, "drive_mode": "api", "drive_mounted": False,
                  "drive_folder": f"My Drive/{cfg['drive_folder']}", "browser_consent_needed": False,
                  "drive_email": about["user"]["emailAddress"]}
        runtime = runtime_info()
        if runtime is not None:
            vm = remote_call("connect", item=runtime)
            if (vm.get("drive_email") or "").lower() != cfg["expected_email"].lower():
                raise RuntimeError("The VM reached a different Drive account; nothing was written.")
            result["vm_connected"] = True
        return result


# -- persistence -----------------------------------------------------------------------
def save_and_stop(*, stop=True):
    with operation_lock():
        item = require_runtime()
        return _save_and_stop(item, stop)


def _save_and_stop(item, stop):
    with token_refresher(item):
        receipt = remote_call("seal", item=item)
    # nothing_to_save: no managed workspace, so no managed data can be lost. Unmanaged
    # files are never captured, as in every other stop.
    nothing = bool(receipt.get("nothing_to_save"))
    if not receipt.get("drive_upload_verified") and not nothing:
        raise RuntimeError("Drive did not verify every checkpoint upload. Shutdown refused.")
    if not nothing:
        remember_pairs(config(), receipt["checkpoints"])
        # Keep a small local receipt for diagnosis; the data itself lives in Drive.
        remote.write_json(CONFIG_DIR / "last-save.json",
                          {**{k: v for k, v in receipt.items() if k != "checkpoints"},
                           "checkpoints": [checkpoint_summary(s) for s in receipt["checkpoints"]]})
        (CONFIG_DIR / "last-save.json").chmod(0o600)
    if stop:
        cfg = config()
        current = require_runtime()
        if colab_api.runtime_id_of(current) != colab_api.runtime_id_of(item):
            raise RuntimeError("Runtime changed during saving. Refusing to stop a different VM.")
        api_for(cfg).delete_runtime(colab_api.runtime_id_of(item))
        close_ssh(item)
        runtimes.forget(CONFIG_DIR, cfg["expected_email"], cfg.get("session", "cuda"),
                        colab_api.runtime_id_of(item))
    return {"stopped": stop, "drive_upload_verified": bool(receipt.get("drive_upload_verified")),
            "drive_flushed": bool(receipt.get("drive_flushed")), "nothing_to_save": nothing,
            "verified_at": receipt.get("verified_at"), "flushed_at": receipt.get("flushed_at"),
            "checkpoints": [checkpoint_summary(s) for s in receipt["checkpoints"]]}


def _pairs_path(cfg):
    return accounts.account_dir(CONFIG_DIR, cfg["expected_email"]) / "verified-pairs.json"


def known_pair(cfg, project):
    """The last checkpoint pair this computer saw verified for a project, if any."""
    try:
        pairs = json.loads(_pairs_path(cfg).read_text())
    except (FileNotFoundError, ValueError):
        return None
    return pairs.get(cfg["drive_folder"], {}).get(project)


def remember_pairs(cfg, checkpoints):
    path = _pairs_path(cfg)
    try:
        pairs = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        pairs = {}
    folder = pairs.setdefault(cfg["drive_folder"], {})
    for item in checkpoints:
        if item.get("manifest_file_id") and item.get("archive_file_id"):
            folder[item["project"]] = {"snapshot": item["snapshot"], "manifest_file_id": item["manifest_file_id"],
                                       "archive_file_id": item["archive_file_id"]}
    accounts._private_write(path, pairs)


def checkpoint_summary(item):
    return {**{k: v for k, v in item.items() if k != "files"}, "file_count": len(item["files"])}


def snapshots(project):
    """List complete checkpoints straight from Drive on this computer; no VM needed."""
    cfg = config()
    drive = local_drive(cfg)
    folder = drive.resolve([cfg["drive_folder"], "projects", remote.valid_name(project)])
    if folder is None:
        return []
    files = drive.children(folder, folders=False)
    archives = {}
    for item in files:
        if item["name"].endswith(".tar.gz"):
            archives.setdefault(item["name"][:-7], []).append(item)
    result, seen = [], set()
    for item in sorted((f for f in files if f["name"].endswith(".json")), key=lambda f: f["name"], reverse=True):
        snapshot = item["name"][:-5]
        props = item.get("appProperties") or {}
        if props.get("cp_kind") == "manifest" and props.get("cp_snapshot") == snapshot:
            summary = {"schema": 1, "project": project, "snapshot": snapshot,
                       "created_at": props.get("cp_created_at"), "reason": props.get("cp_reason"),
                       "archive_bytes": int(props.get("cp_archive_bytes") or 0),
                       "archive_sha256": props.get("cp_archive_sha256"),
                       "file_count": int(props.get("cp_file_count") or 0)}
        else:
            try:
                manifest = json.loads(drive.read_bytes(item["id"]))
                if manifest.get("schema") != 1 or manifest.get("snapshot") != snapshot:
                    continue
                summary = checkpoint_summary(manifest)
            except (ValueError, KeyError, TypeError):
                continue  # A broken marker is not a checkpoint.
        if snapshot not in seen and any(remote.archive_matches(a, summary) for a in archives.get(snapshot, [])):
            seen.add(snapshot)
            result.append(summary)
    return result


def prepare(project="default"):
    with operation_lock():
        project = remote.valid_name(project)
        return remote_call("prepare", project=project, known=known_pair(config(), project))


def restore(project, snapshot="latest"):
    with operation_lock():
        project = remote.valid_name(project)
        return remote_call("restore", project=project, snapshot=snapshot, known=known_pair(config(), project))


def dataset_plan(manifest_path, cache_gib=40, reserve_gib=20):
    """Validate metadata locally, without allocating a GPU or reading shard contents."""
    manifest = datasets.load_manifest(Path(manifest_path).expanduser())
    return datasets.plan(manifest, cache_bytes=cache_gib * remote.GIB, reserve_bytes=reserve_gib * remote.GIB)


def validate_source(script_path, source_directory=None):
    """Check local source paths and sizes without authentication or content reads."""
    script = Path(script_path).expanduser().resolve(strict=True)
    if script.suffix != ".py" or not script.is_file():
        raise ValueError("run accepts a local Python script; it may invoke nvcc or other build tools.")
    source = Path(source_directory).expanduser().resolve(strict=True) if source_directory else script.parent
    if not source.is_dir():
        raise ValueError("Source must be a project directory.")
    if source_directory and source in {Path.home(), Path("/")}:
        raise ValueError("Select a project directory, not your home or filesystem root.")
    try:
        relative = script.relative_to(source)
    except ValueError:
        raise ValueError("Script must be inside the selected source directory.") from None
    if remote.excluded(relative):
        raise ValueError("Script path matches a credential/cache exclusion.")
    limit = config().get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT)
    if script.stat().st_size > limit:
        raise RuntimeError("Script exceeds the source upload size limit.")
    if source_directory:
        total = 0
        for directory, names, files in os.walk(source, followlinks=False):
            names[:] = [name for name in names
                        if not remote.excluded(Path(directory, name).relative_to(source))
                        and not Path(directory, name).is_symlink()]
            for name in files:
                path = Path(directory, name)
                if remote.excluded(path.relative_to(source)) or path.is_symlink():
                    continue
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                total += info.st_size
                if total > limit:
                    raise RuntimeError("Workspace/source exceeds the snapshot size limit. Keep datasets and base "
                                       "model caches outside the workspace; use the bounded dataset cache.")
    return script, source, relative


def validate_run_inputs(script_path, project="default", source_directory=None, checkpoint_seconds=60):
    """Reject local input errors before provisioning or connecting to a runtime."""
    remote.valid_name(project)
    if type(checkpoint_seconds) is not int or checkpoint_seconds < 10:
        raise ValueError("Checkpoint interval must be at least 10 seconds.")
    return validate_source(script_path, source_directory)


def upload_source(item, project, script_path, source_directory=None):
    script, source, relative = validate_source(script_path, source_directory)
    limit = config().get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT)
    files = remote.inventory(source, limit) if source_directory else {relative.as_posix(): {}}
    incoming = f"{ENGINE_ROOT}/source-{uuid.uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="colab-upload-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as target:
            for name in files:
                remote.add_regular(target, source / name, name)
        with archive.open("rb") as stream:
            result = subprocess.run([*ssh_command(item), "cat > " + incoming], stdin=stream,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if result.returncode:
            raise RuntimeError("Could not upload project source: " + result.stderr.decode(errors="replace")[-1000:])
    workspace = "/content/colab-persist/" + project + "/workspace"
    code = ("from pathlib import Path\nfrom colab_persist import remote\n"
            f"try:\n    remote.extract_archive(Path({incoming!r}), Path({workspace!r}), {limit!r})\n"
            f"finally:\n    Path({incoming!r}).unlink(missing_ok=True)\n")
    checked([*ssh_command(item), f"cd {ENGINE_ROOT} && python3 -"], data=code.encode())
    return relative.as_posix()


def run_script(script_path, project="default", arguments=None, source_directory=None,
               checkpoint_seconds=60, stop_after=True, detach=False, log=None):
    validate_run_inputs(script_path, project, source_directory, checkpoint_seconds)
    with operation_lock():
        item = require_runtime()
        prepared = remote_call("prepare", item=item, project=project, known=known_pair(config(), project))
        entry = upload_source(item, project, script_path, source_directory)
        job = remote_call("launch", item=item, project=project, script=entry, arguments=arguments or [],
                          interval=checkpoint_seconds, stop_after=stop_after)
    if job.get("state") == "refused":
        raise RuntimeError(job.get("error") or "The VM refused the job.")
    if detach:
        return {"detached": True, "project": project, "run_id": job.get("run_id"), "state": job.get("state"),
                "restored": prepared["restored"], "next": f"colab-persist attach --project {project}"}
    result = monitor(project, item=item, log=log, run_id=job.get("run_id"))
    return {**result, "restored": prepared["restored"]}


def job_status(project="default", tail=4000):
    """One look at a managed job; also refreshes the VM's Drive token."""
    item = require_runtime()
    return remote_call("job_status", item=item, project=remote.valid_name(project), tail=tail, timeout=120)


def attach(project="default", stop_after=None, log=None):
    item = require_runtime()
    current = remote_call("job_status", item=item, project=remote.valid_name(project), tail=4000, timeout=120)
    if current.get("state") == "none":
        raise RuntimeError(f"No managed job has run in project {project} on this runtime.")
    return monitor(project, item=item, stop_after=stop_after, log=log, offset=current.get("offset", 0),
                   first=current, run_id=current.get("run_id"))


def monitor(project, *, item, stop_after=None, log=None, offset=0, first=None, run_id=None):
    """Follow one detached job (pinned by run ID) without holding the local lock; finalize when it ends."""
    log = log or sys.stderr
    tail, failures, current, announced = "", 0, first, None
    while True:
        if current is None:
            try:
                current = remote_call("job_status", item=item, project=project, offset=offset, timeout=120)
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures += 1
                if failures >= 30:
                    raise RuntimeError("Lost contact with the runtime while following the job. It may still be "
                                       f"running; reconnect with `colab-persist attach --project {project}`. "
                                       + str(error)) from None
                try:
                    present = runtime_info()
                except Exception:
                    present = item  # This computer may be offline; keep retrying.
                if present is None or colab_api.runtime_id_of(present) != colab_api.runtime_id_of(item):
                    raise RuntimeError("The runtime disappeared during the job. Start a new one and restore the "
                                       "last verified checkpoint.") from None
                time.sleep(min(60, 5 * failures))
                continue
        if run_id is None:
            run_id = current.get("run_id")
        if current.get("state") != "none" and current.get("run_id") != run_id:
            raise RuntimeError(f"A different job replaced the one being followed in project {project}; "
                               "not acting on it. Check `colab-persist job --project " + project + "`.")
        if current.get("log"):
            log.write(current["log"])
            log.flush()
            tail = (tail + current["log"])[-8000:]
        offset = current.get("offset", offset)
        state = current.get("state")
        if state == "awaiting_drive" and announced != current.get("error"):
            announced = current.get("error")
            log.write(f"[colab-persist] Waiting to save to Drive: {announced}\n")
            log.flush()
        more = offset < current.get("log_bytes", 0)
        if state in TERMINAL_JOB_STATES and not more:
            break
        current = None
        if not more:
            time.sleep(POLL_SECONDS)
    if state == "none":
        raise RuntimeError(f"No managed job found for project {project}.")
    if state == "refused":
        raise RuntimeError(current.get("error") or "Another managed job holds this project.")
    if state == "lost":
        raise RuntimeError(current.get("error") or "The job worker stopped unexpectedly.")
    if state == "exited_unsaved":
        raise RuntimeError("The script exited but its final checkpoint was not saved: "
                           f"{current.get('error')}. The VM was left running; fix the cause and run "
                           "`colab-persist stop`.")
    stop = current.get("stop_after", True) if stop_after is None else stop_after
    with waiting_lock():
        final = remote_call("job_status", item=item, project=project, tail=0, timeout=120)
        if final.get("run_id") != run_id or final.get("state") != "finished":
            raise RuntimeError(f"Project {project} changed before the final save (another job started); "
                               "nothing was stopped.")
        saved = _save_and_stop(item, stop)
    checkpoint = dict(current.get("checkpoint") or {})
    checkpoint["durability"] = remote.DURABLE
    return {"exit_code": current.get("exit_code"), "checkpoint": checkpoint, "log_tail": tail,
            "periodic_checkpoint_errors": current.get("periodic_checkpoint_errors", []), "save": saved}


# -- one-time import of v0.3 checkpoints ---------------------------------------------------
def legacy_mount(cfg, item):
    """Mount Drive the v0.3 way once, using Google's colab CLI in a temporary environment."""
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("The one-time import needs `uv` (https://docs.astral.sh/uv/).")
    runtime = api_for(cfg).get_runtime(colab_api.runtime_id_of(item))
    if runtime is None:
        raise RuntimeError("The managed runtime disappeared.")
    info = runtime["connectionInfo"]
    directory = accounts.account_dir(CONFIG_DIR, cfg["expected_email"])
    env = os.environ.copy()
    env["GOOGLE_APPLICATION_CREDENTIALS"] = str(accounts.credentials_path(cfg))
    with tempfile.TemporaryDirectory(prefix="legacy-", dir=directory) as temporary:
        arguments = Path(temporary) / "mount.json"
        descriptor = os.open(arguments, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump({"config_path": str(Path(temporary) / "sessions.json"), "name": cfg.get("session", "cuda"),
                       "url": info["url"], "token": info["token"], "expires_at": info.get("expireTime"),
                       "endpoint": colab_api.endpoint_from_url(info["url"])}, handle)
        command = [uv, "run", "--no-project", "--quiet", "--with", LEGACY_CLI, "python",
                   str(Path(__file__).with_name("legacy_mount.py")), str(arguments)]
        for attempt in range(3):
            subprocess.run(command, env=env)
            if remote_call("status", item=item).get("drive_mounted"):
                return True
            if attempt < 2:
                print("Drive did not finish mounting. A 'ValueError: mount failed' just after consent is Colab's "
                      "first mount racing the new credentials. Retrying in 5 s.", file=sys.stderr)
                time.sleep(5)
    raise RuntimeError("The legacy Drive mount did not complete. Run `colab-persist import-legacy` again.")


def import_legacy(projects=None, all_snapshots=False, legacy_folder=None):
    with operation_lock():
        cfg = config()
        item = require_runtime()
        if not remote_call("status", item=item).get("drive_mounted"):
            legacy_mount(cfg, item)
        _TOKENS.clear()  # Start the long transfer with a full-lifetime token.
        with token_refresher(item):
            report = remote_call("import_legacy", item=item, timeout=6 * 3600,
                                 legacy_folder=legacy_folder or cfg["legacy_drive_folder"],
                                 projects=list(projects) if projects else None, all_snapshots=bool(all_snapshots))
        newest = {}
        for entry in report.get("imported", []) + report.get("skipped", []):
            if entry.get("manifest_file_id") and entry["snapshot"] >= newest.get(entry["project"], {}).get("snapshot", ""):
                newest[entry["project"]] = entry
        remember_pairs(cfg, list(newest.values()))
        return report
