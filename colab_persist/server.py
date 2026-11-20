"""Local stdio MCP server. No listening port or credentials in tool results."""
import asyncio
from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from mcp.server.mcpserver.exceptions import ToolError
from . import backend

mcp = MCPServer("Colab Persist")


async def perform(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except (RuntimeError, ValueError, OSError) as error:
        raise ToolError(str(error)) from None


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def runtime_status() -> dict:
    """Check the configured Colab runtime and Drive connection without provisioning compute."""
    return await perform(backend.status)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def runtime_specs() -> dict:
    """List the runtime types (GPU, TPU, CPU, high-memory) this account is eligible for, and its Colab tier."""
    return await perform(backend.specs)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def plan_dataset(manifest_path: str, cache_gib: int = 40, reserve_gib: int = 20) -> dict:
    """Validate a local immutable shard manifest and size a bounded cache for large datasets.

    Reads only the manifest, not the corpus. Does not provision a VM, contact Drive,
    or confirm source files, VM free disk, Drive quota, throughput or GPU model fit.
    Training scripts use colab_persist.datasets.ShardCache on the VM.
    """
    return await perform(backend.dataset_plan, manifest_path, cache_gib, reserve_gib)


@mcp.tool()
async def start_runtime(gpu: str | None = None, tpu: str | None = None, cpu: bool = False,
                        high_mem: bool = False) -> dict:
    """Provision or reuse the configured runtime. Default GPU is L4. Consumes Colab units.

    Drive connects automatically through the Drive API; no browser consent is needed.
    Use runtime_specs to see eligible GPU/TPU types. Choose at most one of gpu, tpu or cpu.
    """
    return await perform(backend.start_runtime, gpu, tpu, cpu, high_mem)


@mcp.tool()
async def prepare_workspace(project: str = "default") -> dict:
    """Create a managed workspace or restore its latest Drive checkpoint on a fresh VM."""
    return await perform(backend.prepare, project)


@mcp.tool()
async def run_script(script_path: str, project: str = "default", arguments: list[str] | None = None,
                     source_directory: str | None = None, checkpoint_seconds: int = 60,
                     stop_after: bool = True, detach: bool = False) -> dict:
    """Run a local Python script on the active GPU VM as a detached managed job.

    Restores previous files, uploads the script (or an explicitly selected source directory),
    checkpoints to Drive periodically with verified uploads, saves on exit, then stops by default.
    The job keeps running on the VM if this call is interrupted. With detach=true the call returns
    right after launch; follow with job_status or attach_job. Only workspace files are saved.
    """
    return await perform(backend.run_script, script_path, project, arguments,
                         source_directory, checkpoint_seconds, stop_after, detach)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def job_status(project: str = "default") -> dict:
    """Check a managed job's state, last checkpoint and recent log lines. Refreshes the VM's Drive token."""
    return await perform(backend.job_status, project)


@mcp.tool()
async def attach_job(project: str = "default", stop_after: bool | None = None) -> dict:
    """Follow a running managed job until it ends, then save and (by default) stop. Can be long-running."""
    return await perform(backend.attach, project, stop_after)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_checkpoints(project: str = "default") -> dict:
    """List complete checkpoints in Drive. Works from this computer without a running VM."""
    return {"checkpoints": await perform(backend.snapshots, project)}


@mcp.tool()
async def restore_workspace(project: str = "default", snapshot: str = "latest") -> dict:
    """Restore a verified Drive checkpoint into an empty workspace. Refuses overwrites."""
    return await perform(backend.restore, project, snapshot)


@mcp.tool()
async def save_workspaces() -> dict:
    """Checkpoint all managed workspaces to Drive with verified uploads. VM stays allocated.

    Refuses if managed jobs are active.
    """
    return await perform(backend.save_and_stop, stop=False)


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def safe_stop() -> dict:
    """Save all managed workspaces and stop the VM only after Drive verifies every upload.

    Unmanaged processes and files outside the workspace are not captured. Saving failures
    leave the VM running so the user can recover. Refuses while managed jobs are active.
    With no managed workspace there is nothing to save; the VM stops and the result
    reports nothing_to_save: true.
    """
    return await perform(backend.save_and_stop, stop=True)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
