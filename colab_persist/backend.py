"""Local control plane. Credentials stay on the Mac; the VM receives only project files."""
from __future__ import annotations

import contextlib
import fcntl
import io
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from google.auth.transport.requests import AuthorizedSession
from colab_cli.auth import AuthProvider, _get_adc_credentials
from colab_cli.common import State
from . import remote

CONFIG_DIR = Path.home() / ".config" / "colab-persist"
REMOTE_ENGINE = "/content/.colab-persist/remote.py"


def config():
    path = CONFIG_DIR / "config.json"
    if not path.exists():
        raise RuntimeError("Configure your account first: colab-persist configure --email YOUR_GOOGLE_EMAIL")
    value = json.loads(path.read_text())
    if not value.get("expected_email"):
        raise RuntimeError("The configuration must specify expected_email.")
    remote.valid_name(value.get("session", "cuda"))
    remote.valid_name(value.get("drive_folder", "Colab-CUDA"))
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
    logging.getLogger("colab_cli.auth").setLevel(logging.ERROR)
    logging.getLogger("google.auth._default").setLevel(logging.ERROR)
    credentials = _get_adc_credentials()
    response = AuthorizedSession(credentials).get(
        "https://openidconnect.googleapis.com/v1/userinfo", timeout=20)
    response.raise_for_status()
    email = response.json().get("email", "").lower()
    if email != cfg["expected_email"].lower():
        raise RuntimeError(f"Google account mismatch: expected {cfg['expected_email']}, got {email}.")
    return email


def colab_command():
    executable = Path(sys.executable).parent / "colab"
    if not executable.exists():
        executable = Path(shutil.which("colab") or "")
    if not executable.is_file():
        raise RuntimeError("Install google-colab-cli in this environment first.")
    return [str(executable), "--auth", "adc"]


def checked(arguments, *, data=None, timeout=180):
    result = subprocess.run(arguments, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout)
    if result.returncode:
        detail = (result.stderr + result.stdout).decode(errors="replace")[-4000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")
    return result.stdout


def session_info():
    identity()
    state = State()
    state.auth_provider = AuthProvider.ADC
    with contextlib.redirect_stdout(sys.stderr):
        sessions, _ = state.sync_sessions()
    item = sessions.get(config().get("session", "cuda"))
    return item


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
    request = {"operation": operation, "folder": config().get("drive_folder", "Colab-CUDA"), **parameters}
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
    request = {"operation": operation, "folder": config().get("drive_folder", "Colab-CUDA"), **parameters}
    code = ("import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('_persist_engine', {REMOTE_ENGINE!r})\n"
            "_engine = importlib.util.module_from_spec(_spec)\n_spec.loader.exec_module(_engine)\n"
            f"_engine.dispatch({request!r})\n")
    with tempfile.TemporaryDirectory(prefix="colab-persist-") as temporary:
        script = Path(temporary) / "operation.py"
        script.write_text(code)
        raw = checked([*colab_command(), "exec", "-s", item.name, "-f", str(script),
                       "--timeout", str(timeout)], timeout=timeout + 60).decode(errors="replace")
    return parse_result(raw)


def install_engine(item):
    source = Path(remote.__file__).read_bytes()
    for attempt in range(3):
        try:
            checked([*ssh_command(item), "mkdir -p /content/.colab-persist && cat > " + REMOTE_ENGINE], data=source)
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
                     "--gpu", selected], timeout=300)
            item = require_session()
        install_engine(item)
        write_ssh_alias(item)
        return {"session": item.name, "endpoint": item.endpoint, "gpu": item.accelerator, "created": created,
                **remote_call("status", item=item)}


def status():
    item = session_info()
    if item is None:
        return {"active": False, "account_verified": True, "next": "colab-persist start"}
    return {"active": True, "session": item.name, "endpoint": item.endpoint,
            **remote_call("status", item=item)}


def mount_drive():
    """Terminal-only consent flow. Never read OAuth codes through MCP/chat."""
    with operation_lock():
        item = require_session()
        if remote_call("status", item=item)["drive_mounted"]:
            return {"drive_mounted": True}
        for attempt in range(2):
            completed = subprocess.run([*colab_command(), "drivemount", "-s", item.name])
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
        checked([*colab_command(), "stop", "-s", item.name], timeout=180)
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


def upload_source(item, project, script_path, source_directory=None):
    script = Path(script_path).expanduser().resolve(strict=True)
    if script.suffix != ".py" or not script.is_file():
        raise ValueError("run accepts a local Python script; it may invoke nvcc or other build tools.")
    source = Path(source_directory).expanduser().resolve(strict=True) if source_directory else script.parent
    if source_directory and source in {Path.home(), Path("/")}:
        raise ValueError("Select a project directory, not your home or filesystem root.")
    relative = script.relative_to(source)
    if remote.excluded(relative):
        raise ValueError("Script path matches a credential/cache exclusion.")
    files = remote.inventory(source) if source_directory else {relative.as_posix(): {}}
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
            f"m.extract_archive(m.Path({incoming!r}),m.Path({workspace!r}))\n")
    checked([*ssh_command(item), "python3 -"], data=code.encode())
    return relative.as_posix()


def run_script(script_path, project="default", arguments=None, source_directory=None,
               checkpoint_seconds=60, stop_after=True):
    remote.valid_name(project)
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
