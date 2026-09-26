import sys
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
        self.assertEqual(len(tools), 8)
        self.assertTrue(tools["runtime_status"].annotations.read_only_hint)
        self.assertTrue(tools["safe_stop"].annotations.destructive_hint)
        self.assertIn("script_path", tools["run_script"].input_schema["required"])

    async def test_expected_failure_is_a_readable_tool_error(self):
        with patch("colab_persist.backend.restore", side_effect=RuntimeError("Checksum mismatch; restore aborted")):
            async with Client(mcp) as client:
                result = await client.call_tool("restore_workspace", {"project": "demo"})
        self.assertTrue(result.is_error)
        self.assertIn("Checksum mismatch", result.content[0].text)
