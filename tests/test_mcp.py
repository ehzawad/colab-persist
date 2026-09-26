import sys
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mcp import Client, StdioServerParameters
from colab_persist.server import mcp


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_tool_discovery(self):
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "colab_persist.server"])
        async with Client(parameters) as client:
            response = await client.list_tools()
        tools = {tool.name: tool for tool in response.tools}
        self.assertEqual(len(tools), 9)
        self.assertTrue(tools["runtime_status"].annotations.read_only_hint)
        self.assertTrue(tools["safe_stop"].annotations.destructive_hint)
        self.assertIn("script_path", tools["run_script"].input_schema["required"])
        self.assertTrue(tools["plan_dataset"].annotations.read_only_hint)

    async def test_terabyte_plan_reads_only_metadata_without_allocating_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory).resolve() / "corpus.json"
            manifest.write_text(json.dumps({"schema": 1, "name": "agent-trajectories", "version": "v1",
                "shards": [{"path": f"v1/{i:04d}.parquet", "size": 1_000_000_000,
                            "sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(1000)]}))
            with patch("colab_persist.backend.require_session") as runtime:
                async with Client(mcp) as client:
                    result = await client.call_tool("plan_dataset", {"manifest_path": str(manifest)})
                self.assertFalse(result.is_error)
                payload = result.structured_content or json.loads(result.content[0].text)
                self.assertEqual(payload["total_bytes"], 1_000_000_000_000)
                self.assertEqual(payload["cache_bytes"], 40 * 1024**3)
                runtime.assert_not_called()

    async def test_expected_failure_is_a_readable_tool_error(self):
        with patch("colab_persist.backend.restore", side_effect=RuntimeError("Checksum mismatch; restore aborted")):
            async with Client(mcp) as client:
                result = await client.call_tool("restore_workspace", {"project": "demo"})
        self.assertTrue(result.is_error)
        self.assertIn("Checksum mismatch", result.content[0].text)
