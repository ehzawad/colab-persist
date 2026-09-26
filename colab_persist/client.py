"""Human-friendly terminal client backed by the same stdio MCP tools agents use."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from mcp import Client, StdioServerParameters
from . import accounts, backend, remote


async def call_tool(name, arguments):
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "colab_persist.server"])
    async with Client(parameters, read_timeout_seconds=90000) as client:
        result = await client.call_tool(name, arguments, read_timeout_seconds=90000)
    if result.is_error:
        raise RuntimeError("\n".join(item.text for item in result.content if hasattr(item, "text")))
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(next(item.text for item in result.content if hasattr(item, "text")))


def invoke(name, **arguments):
    return asyncio.run(call_tool(name, arguments))


def configure(args):
    with backend.operation_lock():
        return _configure(args)


def _configure(args):
    email = accounts.normalize_email(args.email)
    if args.snapshot_limit_gib is not None and args.snapshot_limit_gib <= 0:
        raise ValueError("Snapshot limit must be positive.")
    backend.CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = backend.CONFIG_DIR / "config.json"
    previous = json.loads(target.read_text()) if target.exists() else {}
    if previous and email != previous.get("expected_email", "").lower():
        raise ValueError("configure does not switch Google credentials. Use `colab-persist login --email "
                         + email + "` to verify and switch accounts safely.")
    session = args.session or previous.get("session", "cuda")
    gpu = args.gpu or previous.get("gpu", "L4")
    remote.valid_name(session)
    key = Path(args.key or previous.get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser().absolute()
    if previous and (session != previous.get("session", "cuda") or
                     str(key) != str(Path(previous.get("ssh_identity", "~/.ssh/id_ed25519_colab")).expanduser().absolute())):
        if backend.assignments_for(previous):
            raise RuntimeError("Save and stop active runtimes before changing the session name or SSH key.")
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "colab-persist",
                        "-f", str(key)], check=True, stdout=subprocess.DEVNULL)
    key.chmod(0o600)
    value = {**previous, "expected_email": email, "session": session,
             "gpu": gpu, "ssh_identity": str(key), "drive_folder": previous.get("drive_folder", "Colab-CUDA")}
    if args.snapshot_limit_gib is not None:
        value["snapshot_limit_bytes"] = args.snapshot_limit_gib * remote.GIB
    if args.ssh_config:
        value["ssh_alias_config"] = str(Path(args.ssh_config).expanduser().absolute())
    remote.write_json(target, value)
    target.chmod(0o600)
    return {"configured": True, "config": str(target), "default_gpu": gpu}


def main():
    parser = argparse.ArgumentParser(description="Restore, run, checkpoint to Google Drive, then safely stop Colab.")
    commands = parser.add_subparsers(dest="command", required=True)
    cfg = commands.add_parser("configure", help="Set the expected Google account and GPU preference")
    cfg.add_argument("--email", required=True)
    cfg.add_argument("--gpu", choices=["T4", "L4", "G4", "H100", "A100"], help="Preserve existing setting; initially L4")
    cfg.add_argument("--session", help="Preserve existing setting; initially cuda")
    cfg.add_argument("--key", help="Preserve existing key; initially ~/.ssh/id_ed25519_colab")
    cfg.add_argument("--ssh-config", help="Optional dedicated SSH config fragment to manage; include it from ~/.ssh/config")
    cfg.add_argument("--snapshot-limit-gib", type=int,
                     help="Workspace/source size cap before hashing (default 5 GiB); excludes external caches")
    login = commands.add_parser("login", help="Verify and switch Google accounts using private credentials")
    login.add_argument("--email", required=True)
    login.add_argument("--reauth", action="store_true", help="Refresh saved login through browser consent")
    login.add_argument("--no-launch-browser", action="store_true", help="Print the consent URL for manual browser login")
    start = commands.add_parser("start", help="Start/reuse the selected GPU and mount Drive")
    start.add_argument("--gpu", choices=["T4", "L4", "G4", "H100", "A100"])
    start.add_argument("--no-mount", action="store_true")
    start.add_argument("--project", default="default")
    commands.add_parser("mount", help="Authorize and mount Drive in your terminal")
    commands.add_parser("status")
    commands.add_parser("sessions", help="List all runtimes on the selected Google account, including unmanaged ones")
    commands.add_parser("save", help="Save all managed workspaces and flush/unmount Drive; keep the VM")
    commands.add_parser("stop", help="Save and flush Drive before stopping the VM")
    commands.add_parser("tools", help="List the real MCP tools")
    dataset = commands.add_parser("dataset-plan", help="Validate a shard manifest and cache budget locally; no GPU allocation")
    dataset.add_argument("manifest")
    dataset.add_argument("--cache-gib", type=int, default=40)
    dataset.add_argument("--reserve-gib", type=int, default=20)
    shell = commands.add_parser("ssh", help="Open SSH or run a remote command; no tmux")
    shell.add_argument("remote_command", nargs=argparse.REMAINDER)
    snapshots = commands.add_parser("snapshots")
    snapshots.add_argument("--project", default="default")
    restore = commands.add_parser("restore")
    restore.add_argument("--project", default="default")
    restore.add_argument("--snapshot", default="latest")
    run = commands.add_parser("run", help="Run a local Python script; arguments after -- go to your script")
    run.add_argument("script")
    run.add_argument("--project", default="default")
    run.add_argument("--source", help="Explicit project directory to upload; otherwise upload only the script")
    run.add_argument("--gpu", choices=["T4", "L4", "G4", "H100", "A100"])
    run.add_argument("--checkpoint-seconds", type=int, default=60)
    run.add_argument("--keep-runtime", action="store_true", help="Flush Drive but leave the GPU allocated")
    argv = sys.argv[1:]
    forwarded = []
    if argv and argv[0] == "run" and "--" in argv:
        cut = argv.index("--")
        forwarded, argv = argv[cut + 1:], argv[:cut]
    args = parser.parse_args(argv)
    try:
        if args.command == "configure":
            result = configure(args)
        elif args.command == "login":
            result = backend.login(args.email, reauth=args.reauth, no_launch_browser=args.no_launch_browser)
        elif args.command == "sessions":
            result = backend.list_sessions()
        elif args.command == "mount":
            result = backend.mount_drive()
        elif args.command == "dataset-plan":
            result = invoke("plan_dataset", manifest_path=str(Path(args.manifest).expanduser().absolute()),
                            cache_gib=args.cache_gib, reserve_gib=args.reserve_gib)
        elif args.command in {"start", "run"}:
            if args.command == "run":
                backend.validate_run_inputs(args.script, args.project, args.source,
                                            args.checkpoint_seconds)
            else:
                remote.valid_name(args.project)
            result = invoke("start_runtime", gpu=args.gpu)
            print(f"Runtime {result['session']}: {result['gpu']}", file=sys.stderr)
            if args.command == "run" or not args.no_mount:
                backend.mount_drive()
            if args.command == "run":
                print("Running remotely; workspace checkpoints will be saved to Drive.", file=sys.stderr)
                result = invoke("run_script", script_path=str(Path(args.script).expanduser().absolute()),
                                project=args.project, arguments=forwarded,
                                source_directory=str(Path(args.source).expanduser().absolute()) if args.source else None,
                                checkpoint_seconds=args.checkpoint_seconds, stop_after=not args.keep_runtime)
            else:
                result = invoke("runtime_status")
                if result["drive_mounted"]:
                    result["prepared"] = invoke("prepare_workspace", project=args.project)
        elif args.command == "ssh":
            item = backend.require_session()
            command = backend.ssh_command(item)
            rest = args.remote_command
            if rest and rest[0] == "--":
                rest = rest[1:]
            if rest:
                command += [shlex.join(rest)]
            else:
                command.insert(1, "-t")
            raise SystemExit(subprocess.call(command))
        elif args.command == "tools":
            async def list_tools():
                params = StdioServerParameters(command=sys.executable, args=["-m", "colab_persist.server"])
                async with Client(params) as client:
                    response = await client.list_tools()
                    return [tool.name for tool in response.tools]
            result = asyncio.run(list_tools())
        else:
            tool, parameters = {
                "status": ("runtime_status", {}), "save": ("save_workspaces", {}),
                "stop": ("safe_stop", {}),
                "snapshots": ("list_checkpoints", {"project": getattr(args, "project", "default")}),
                "restore": ("restore_workspace", {"project": getattr(args, "project", "default"),
                                                   "snapshot": getattr(args, "snapshot", "latest")}),
            }[args.command]
            result = invoke(tool, **parameters)
        print(json.dumps(result, indent=2))
        if args.command == "run" and result.get("exit_code", 0):
            raise SystemExit(result["exit_code"])
    except KeyboardInterrupt:
        print("Interrupted locally. The remote job may still be running; check status before taking action.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"colab-persist: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
