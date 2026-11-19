import sys
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mcp import Client, StdioServerParameters
from colab_persist.server import mcp

TOOLS = {"runtime_status", "runtime_specs", "plan_dataset", "start_runtime", "prepare_workspace", "run_script",
         "job_status", "attach_job", "list_checkpoints", "restore_workspace", "save_workspaces", "safe_stop"}
READ_ONLY = {"runtime_status", "runtime_specs", "plan_dataset", "list_checkpoints", "job_status"}


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_tool_discovery(self):
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "colab_persist.server"])
        async with Client(parameters) as client:
            response = await client.list_tools()
        tools = {tool.name: tool for tool in response.tools}
        self.assertEqual(set(tools), TOOLS)
        for name, tool in tools.items():
            with self.subTest(tool=name):
                read_only = bool(tool.annotations and tool.annotations.read_only_hint)
                self.assertEqual(read_only, name in READ_ONLY)
                destructive = bool(tool.annotations and tool.annotations.destructive_hint)
                self.assertEqual(destructive, name == "safe_stop")
        self.assertIn("script_path", tools["run_script"].input_schema["required"])
        self.assertIn("detach", tools["run_script"].input_schema["properties"])
        self.assertEqual(set(tools["start_runtime"].input_schema["properties"]), {"gpu", "tpu", "cpu", "high_mem"})
        self.assertIn("stop_after", tools["attach_job"].input_schema["properties"])

    async def test_terabyte_plan_reads_only_metadata_without_allocating_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory).resolve() / "corpus.json"
            manifest.write_text(json.dumps({"schema": 1, "name": "agent-trajectories", "version": "v1",
                "shards": [{"path": f"v1/{i:04d}.parquet", "size": 1_000_000_000,
                            "sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(1000)]}))
            with patch("colab_persist.backend.require_runtime") as runtime, \
                 patch("colab_persist.backend.config") as config:
                async with Client(mcp) as client:
                    result = await client.call_tool("plan_dataset", {"manifest_path": str(manifest)})
                self.assertFalse(result.is_error)
                payload = result.structured_content or json.loads(result.content[0].text)
                self.assertEqual(payload["total_bytes"], 1_000_000_000_000)
                self.assertEqual(payload["cache_bytes"], 40 * 1024**3)
                runtime.assert_not_called()
                config.assert_not_called()

    async def test_expected_failure_is_a_readable_tool_error(self):
        with patch("colab_persist.backend.restore", side_effect=RuntimeError("Checksum mismatch; restore aborted")):
            async with Client(mcp) as client:
                result = await client.call_tool("restore_workspace", {"project": "demo"})
        self.assertTrue(result.is_error)
        self.assertIn("Checksum mismatch", result.content[0].text)

    async def test_tools_route_to_the_backend(self):
        with patch("colab_persist.backend.snapshots", return_value=[{"snapshot": "s1"}]) as snapshots, \
             patch("colab_persist.backend.job_status", return_value={"state": "running"}) as job_status, \
             patch("colab_persist.backend.attach", return_value={"exit_code": 0}) as attach, \
             patch("colab_persist.backend.specs", return_value={"eligible": ["L4"]}) as specs, \
             patch("colab_persist.backend.save_and_stop", return_value={"stopped": True}) as save, \
             patch("colab_persist.backend.start_runtime", return_value={"runtime": "cp-x"}) as start:
            async with Client(mcp) as client:
                listed = await client.call_tool("list_checkpoints", {"project": "demo"})
                await client.call_tool("job_status", {"project": "demo"})
                await client.call_tool("attach_job", {"project": "demo", "stop_after": False})
                await client.call_tool("runtime_specs", {})
                await client.call_tool("save_workspaces", {})
                await client.call_tool("safe_stop", {})
                await client.call_tool("start_runtime", {"tpu": "V5E1", "high_mem": True})
        payload = listed.structured_content or json.loads(listed.content[0].text)
        self.assertEqual(payload, {"checkpoints": [{"snapshot": "s1"}]})
        snapshots.assert_called_once_with("demo")
        job_status.assert_called_once_with("demo")
        attach.assert_called_once_with("demo", False)
        specs.assert_called_once_with()
        self.assertEqual([c.kwargs for c in save.call_args_list], [{"stop": False}, {"stop": True}])
        start.assert_called_once_with(None, "V5E1", False, True)


if __name__ == "__main__":
    unittest.main()
