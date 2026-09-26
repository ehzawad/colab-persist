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
    """Check the configured Colab runtime and Drive mount without provisioning compute."""
    return await perform(backend.status)


@mcp.tool()
async def start_runtime(gpu: str | None = None) -> dict:
    """Provision or reuse the configured runtime. Default GPU is L4. Consumes Colab units.

    If drive_mounted is false, run colab-persist mount in a user terminal to authorize Drive.
    """
    return await perform(backend.start_runtime, gpu)


@mcp.tool()
async def prepare_workspace(project: str = "default") -> dict:
    """Create a managed workspace or restore its latest Drive snapshot on a fresh VM."""
    return await perform(backend.prepare, project)


@mcp.tool()
async def run_script(script_path: str, project: str = "default", arguments: list[str] | None = None,
                     source_directory: str | None = None, checkpoint_seconds: int = 60,
                     stop_after: bool = True) -> dict:
    """Run a local Python script on the active, Drive-mounted GPU VM.

    Restores previous files, uploads the script (or an explicitly selected source directory),
    periodically checkpoints, flushes Drive on completion, then stops by default. Script may
    invoke nvcc and other tools. Only workspace files are saved, not RAM or credentials.
    Google may still terminate the VM; completed snapshots are recoverable. Can be long-running.
    """
    return await perform(backend.run_script, script_path, project, arguments,
                                   source_directory, checkpoint_seconds, stop_after)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_checkpoints(project: str = "default") -> dict:
    """List complete checkpoint manifests from the mounted Drive."""
    return {"checkpoints": await perform(backend.snapshots, project)}


@mcp.tool()
async def restore_workspace(project: str = "default", snapshot: str = "latest") -> dict:
    """Restore a verified Drive checkpoint into an empty workspace. Refuses overwrites."""
    return await perform(backend.restore, project, snapshot)


@mcp.tool()
async def save_workspaces() -> dict:
    """Checkpoint all managed workspaces, flush Drive, and unmount it. VM stays allocated.

    Refuses if managed jobs are active. Remount Drive before further persistence operations.
    """
    return await perform(backend.save_and_stop, stop=False)


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def safe_stop() -> dict:
    """Save all managed workspaces and stop the VM only after Drive confirms its flush.

    Unmanaged processes and files outside the workspace are not captured. Saving failures
    leave the VM running so the user can recover. Refuses while managed jobs are active.
    """
    return await perform(backend.save_and_stop, stop=True)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
