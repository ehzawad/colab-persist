"""Local control plane. Credentials stay on the Mac; the VM receives only project files."""
from __future__ import annotations

import contextlib
import fcntl
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
import time

from google.auth.transport.requests import AuthorizedSession
from colab_cli.client import Client, Prod
from . import accounts, datasets, remote, session_state

CONFIG_DIR = Path.home() / ".config" / "colab-persist"
REMOTE_ENGINE = "/content/.colab-persist/remote.py"


def config():
    path = CONFIG_DIR / "config.json"
    if not path.exists():
        raise RuntimeError("Configure your account first: colab-persist configure --email YOUR_GOOGLE_EMAIL")
    value = json.loads(path.read_text())
    if not value.get("expected_email"):
        raise RuntimeError("The configuration must specify expected_email.")
    accounts.normalize_email(value["expected_email"])
    remote.valid_name(value.get("session", "cuda"))
    remote.valid_name(value.get("drive_folder", "Colab-CUDA"))
    limit = value.get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT)
    if type(limit) is not int or limit <= 0:
        raise ValueError("snapshot_limit_bytes must be a positive integer.")
    return value


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


def identity():
    cfg = config()
    accounts.verified_credentials(cfg)
    return cfg["expected_email"].lower()


def assignments_for(cfg):
    credentials = accounts.verified_credentials(cfg)
    with AuthorizedSession(credentials) as session:
        return Client(Prod(), session).list_assignments()


def login(email, *, reauth=False, no_launch_browser=False):
    """Stage, verify and commit account selection under the persistence lock."""
    email = accounts.normalize_email(email)
    with operation_lock():
        previous = config()
        switching = email != previous["expected_email"].lower()

        def require_old_account_stopped():
            if switching and assignments_for(previous):
                raise RuntimeError("The current Google account still has active Colab runtimes. "
                                   "Save and stop them before switching accounts.")

        require_old_account_stopped()
        directory = session_state.session_path(CONFIG_DIR, email).parent
        saved = directory / "credentials.json"
        candidate = {**previous, "expected_email": email, "credentials_file": str(saved)}
        if reauth or not saved.exists():
            # Isolate gcloud's ADC output and configuration; the user's normal ADC
            # and gcloud account selection are never overwritten.
            with tempfile.TemporaryDirectory(prefix="login-", dir=directory) as temporary:
                env = os.environ.copy()
                env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
                env["CLOUDSDK_CONFIG"] = temporary
                command = ["gcloud", "auth", "application-default", "login", email,
                           "--disable-quota-project", "--scopes=" + ",".join(accounts.SCOPES)]
                if no_launch_browser:
                    command.append("--no-launch-browser")
                result = subprocess.run(command, env=env)
                if result.returncode:
                    raise RuntimeError("Google login did not finish; account selection is unchanged.")
                staged = Path(temporary) / "application_default_credentials.json"
                candidate["credentials_file"] = str(staged)
                assignments = assignments_for(candidate)  # Verify email and Colab access before committing.
                require_old_account_stopped()  # A runtime may have started during browser consent.
                staged.chmod(0o600)
                staged.replace(saved)
        else:
            assignments = assignments_for(candidate)
            require_old_account_stopped()
        candidate["credentials_file"] = str(saved)
        remote.write_json(CONFIG_DIR / "config.json", candidate)
        (CONFIG_DIR / "config.json").chmod(0o600)
        return {"account": email, "account_verified": True, "switched": switching,
                "active_runtimes": len(assignments), "global_gcloud_credentials_changed": False,
                "next": "colab-persist status"}


def colab_command():
    executable = Path(sys.executable).parent / "colab"
    if not executable.exists():
        executable = Path(shutil.which("colab") or "")
    if not executable.is_file():
        raise RuntimeError("Install google-colab-cli in this environment first.")
    return [str(executable), "--auth", "adc", "--config",
            str(session_state.session_path(CONFIG_DIR, config()["expected_email"]))]


def checked(arguments, *, data=None, timeout=180, env=None):
    result = subprocess.run(arguments, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, env=env)
    if result.returncode:
        detail = (result.stderr + result.stdout).decode(errors="replace")[-4000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")
    return result.stdout


def session_info():
    cfg = config()
    credentials = accounts.verified_credentials(cfg)
    sessions, _ = session_state.synced_state(CONFIG_DIR, cfg, credentials)
    return sessions.get(cfg.get("session", "cuda"))


def list_sessions():
    with operation_lock():
        cfg = config()
        credentials = accounts.verified_credentials(cfg)
        sessions, assignments = session_state.synced_state(CONFIG_DIR, cfg, credentials)
        names = {item.endpoint: name for name, item in sessions.items()}
        return {"account": cfg["expected_email"], "account_verified": True,
                "active_runtimes": len(assignments), "sessions": [
                    {"endpoint": item.endpoint, "name": names.get(item.endpoint),
                     "managed": names.get(item.endpoint) == cfg.get("session", "cuda")}
                    for item in assignments]}


def require_session():
    item = session_info()
    if item is None:
        raise RuntimeError("No managed runtime is active. Run `colab-persist start`.")
    return item


def ssh_command(item):
    cfg = config()
    key = str(Path(cfg.get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser())
    proxy = shlex.join([sys.executable, "-m", "colab_persist.proxy", "--endpoint", item.endpoint])
    return ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i", key,
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
            "-o", "UserKnownHostsFile=" + str(Path.home() / ".ssh/known_hosts_colab_persist"),
            "-o", "HostKeyAlias=colab-" + item.endpoint,
            "-o", "ProxyCommand=" + proxy, "root@colab-runtime"]


def remote_call(operation, *, item=None, **parameters):
    item = item or require_session()
    request = {"operation": operation, "folder": config().get("drive_folder", "Colab-CUDA"),
               "snapshot_limit": config().get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT), **parameters}
    raw = checked([*ssh_command(item), "python3 " + REMOTE_ENGINE],
                  data=json.dumps(request).encode(), timeout=900).decode(errors="replace")
    return parse_result(raw)


def parse_result(raw):
    results = [line[len(remote.RESULT_PREFIX):] for line in raw.splitlines()
               if line.startswith(remote.RESULT_PREFIX)]
    if not results:
        raise RuntimeError("No confirmed completion receipt was returned. Runtime was NOT stopped.\n" + raw[-2000:])
    return json.loads(results[-1])


def kernel_call(operation, *, item, timeout=900, **parameters):
    request = {"operation": operation, "folder": config().get("drive_folder", "Colab-CUDA"),
               "snapshot_limit": config().get("snapshot_limit_bytes", remote.DEFAULT_SNAPSHOT_LIMIT), **parameters}
    code = ("import importlib.util\n"
            f"import sys; sys.path.insert(0, {str(Path(REMOTE_ENGINE).parent)!r})\n"
            f"_spec = importlib.util.spec_from_file_location('_persist_engine', {REMOTE_ENGINE!r})\n"
            "_engine = importlib.util.module_from_spec(_spec)\n_spec.loader.exec_module(_engine)\n"
            f"_engine.dispatch({request!r})\n")
    with tempfile.TemporaryDirectory(prefix="colab-persist-") as temporary:
        script = Path(temporary) / "operation.py"
        script.write_text(code)
        raw = checked([*colab_command(), "exec", "-s", item.name, "-f", str(script),
                       "--timeout", str(timeout)], timeout=timeout + 60,
                      env=accounts.environment(config())).decode(errors="replace")
    return parse_result(raw)


def install_engine(item):
    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w") as archive:
        archive.add(remote.__file__, arcname="remote.py")
        package = Path(remote.__file__).parent
        for name in ("__init__.py", "remote.py", "datasets.py", "checkpoints.py"):
            archive.add(package / name, arcname="colab_persist/" + name)
    for attempt in range(3):
        try:
            checked([*ssh_command(item), "mkdir -p /content/.colab-persist && tar -xf - -C /content/.colab-persist"],
                    data=source.getvalue())
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(2)
    # SSH shells do not inherit the notebook's CUDA library path.
    setup = b'''set -e
cat > /etc/profile.d/colab-cuda.sh <<'CUDA'
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64-nvidia:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/content/.colab-persist${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/content/colab-persist-cache/huggingface}"
CUDA
python3 - <<'PY'
from pathlib import Path
p = Path.home() / '.bashrc'
line = 'source /etc/profile.d/colab-cuda.sh\\n'
old = p.read_text() if p.exists() else ''
if line not in old:
    p.write_text(line + old)
PY
'''
    checked([*ssh_command(item), "bash -s"], data=setup)


def write_ssh_alias(item):
    destination = config().get("ssh_alias_config")
    if not destination:
        return
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    proxy = shlex.join([sys.executable, "-m", "colab_persist.proxy", "--endpoint", item.endpoint])
    key = Path(config().get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser()
    target.write_text(f'''# Managed by colab-persist start. Endpoint-specific host key pinning.
Host colab-cuda
  HostName colab-runtime
  HostKeyAlias colab-{item.endpoint}
  User root
  IdentityFile "{key}"
  IdentitiesOnly yes
  ProxyCommand {proxy}
  StrictHostKeyChecking accept-new
  UserKnownHostsFile ~/.ssh/known_hosts_colab_persist
  ServerAliveInterval 30
  ServerAliveCountMax 3
  ForwardAgent no
''')
    target.chmod(0o600)


def start_runtime(gpu=None):
    with operation_lock():
        selected = gpu or config().get("gpu", "L4")
        if selected not in {"T4", "L4", "G4", "H100", "A100"}:
            raise ValueError("GPU must be T4, L4, G4, H100 or A100.")
        item = session_info()
        created = item is None
        if item is not None and item.accelerator.upper() != selected.upper():
            raise RuntimeError(f"Runtime already uses {item.accelerator}. Save and stop it before selecting {selected}.")
        if created:
            cfg = config()
            checked([*colab_command(), "new", "-s", cfg.get("session", "cuda"),
                     "--gpu", selected], timeout=300, env=accounts.environment(cfg))
            item = require_session()
        install_engine(item)
        write_ssh_alias(item)
        return {"session": item.name, "endpoint": item.endpoint, "gpu": item.accelerator, "created": created,
                **remote_call("status", item=item)}


def status():
    with operation_lock():
        cfg = config()
        credentials = accounts.verified_credentials(cfg)
        sessions, assignments = session_state.synced_state(CONFIG_DIR, cfg, credentials)
        item = sessions.get(cfg.get("session", "cuda"))
        summary = {"account": cfg["expected_email"], "account_verified": True,
                   "active_runtimes": len(assignments)}
        if item is None:
            return {**summary, "active": False,
                    "next": "colab-persist sessions" if assignments else "colab-persist start"}
        return {**summary, "active": True,
                "session": item.name, "endpoint": item.endpoint, **remote_call("status", item=item)}


def mount_drive():
    """Terminal-only consent flow. Never read OAuth codes through MCP/chat."""
    with operation_lock():
        item = require_session()
        if remote_call("status", item=item)["drive_mounted"]:
            return {"drive_mounted": True}
        for attempt in range(2):
            completed = subprocess.run([*colab_command(), "drivemount", "-s", item.name],
                                       env=accounts.environment(config()))
            if completed.returncode == 0 and remote_call("status", item=item)["drive_mounted"]:
                return {"drive_mounted": True}
            if attempt == 0:
                print("Drive did not finish mounting; retrying once using the completed authorization.", file=sys.stderr)
        raise RuntimeError("Drive mount did not complete. Run `colab-persist mount` again.")


def save_and_stop(*, stop=True):
    with operation_lock():
        item = require_session()
        return _save_and_stop(item, stop)


def _save_and_stop(item, stop):
    receipt = kernel_call("seal", item=item)
    if not receipt.get("drive_flushed"):
        raise RuntimeError("Drive did not confirm its flush. Shutdown refused.")
    # Keep a small local receipt for diagnosis; the data itself lives in Drive.
    remote.write_json(CONFIG_DIR / "last-save.json", receipt)
    (CONFIG_DIR / "last-save.json").chmod(0o600)
    if stop:
        if require_session().endpoint != item.endpoint:
            raise RuntimeError("Runtime changed during saving. Refusing to stop a different VM.")
        checked([*colab_command(), "stop", "-s", item.name], timeout=180,
                env=accounts.environment(config()))
    return {"stopped": stop, "drive_flushed": True, "flushed_at": receipt["flushed_at"],
            "checkpoints": [checkpoint_summary(s) for s in receipt["checkpoints"]]}


def checkpoint_summary(item):
    return {**{k: v for k, v in item.items() if k != "files"}, "file_count": len(item["files"])}


def snapshots(project):
    return [checkpoint_summary(s) for s in remote_call("snapshots", project=remote.valid_name(project))]


def prepare(project="default"):
    with operation_lock():
        return remote_call("prepare", project=remote.valid_name(project))


def restore(project, snapshot="latest"):
    with operation_lock():
        return remote_call("restore", project=remote.valid_name(project), snapshot=snapshot)


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
    with tempfile.TemporaryDirectory(prefix="colab-upload-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as target:
            for name in files:
                target.add(source / name, arcname=name, recursive=False)
        incoming = "/content/.colab-persist/source.tar.gz"
        with archive.open("rb") as stream:
            result = subprocess.run([*ssh_command(item), "cat > " + incoming], stdin=stream,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if result.returncode:
            raise RuntimeError("Could not upload project source: " + result.stderr.decode(errors="replace")[-1000:])
    workspace = "/content/colab-persist/" + project + "/workspace"
    code = ("import importlib.util\n"
            f"s=importlib.util.spec_from_file_location('engine',{REMOTE_ENGINE!r})\n"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
            f"m.extract_archive(m.Path({incoming!r}),m.Path({workspace!r}),{limit!r})\n")
    checked([*ssh_command(item), "python3 -"], data=code.encode())
    return relative.as_posix()


def run_script(script_path, project="default", arguments=None, source_directory=None,
               checkpoint_seconds=60, stop_after=True):
    validate_run_inputs(script_path, project, source_directory, checkpoint_seconds)
    with operation_lock():
        item = require_session()
        prepared = remote_call("prepare", item=item, project=project)
        entry = upload_source(item, project, script_path, source_directory)
        result = kernel_call("run", item=item, timeout=86400, project=project, script=entry,
                             arguments=arguments or [], interval=checkpoint_seconds)
        # Even a script returning nonzero gets its outputs saved. Saving errors never trigger stop.
        saved = _save_and_stop(item, stop_after)
        result["checkpoint"] = checkpoint_summary(result["checkpoint"])
        result["checkpoint"]["durability"] = "drive_flush_confirmed"
        return {**result, "restored": prepared["restored"], "save": saved}
