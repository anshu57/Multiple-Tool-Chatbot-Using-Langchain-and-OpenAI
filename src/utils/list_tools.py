#!/usr/bin/env python3.12
"""
Script to list all available tools (local and MCP).
Usage: python3.12 list_tools.py [--mcp] [--details]
"""

import asyncio
import sys
from tools_manager import ToolsManager


async def main():
    enable_mcp = "--mcp" in sys.argv
    show_details = "--details" in sys.argv

    print("=" * 80)
    print("AVAILABLE TOOLS")
    print("=" * 80)

    # Create tools manager
    tm = await ToolsManager.create(enable_mcp=enable_mcp)

    try:
        # Get all tools
        tools = await tm.get_all_tools()

        if not tools:
            print("No tools available.")
            return

        print(f"\nTotal Tools: {len(tools)}\n")

        # Categorize tools
        local_count = 0
        mcp_count = 0

        for i, tool in enumerate(tools, 1):
            tool_name = getattr(tool, "name", type(tool).__name__)
            tool_type = type(tool).__name__
            description = getattr(tool, "description", "No description")

            # Determine if tool is local or MCP
            is_mcp = "StructuredTool" in tool_type
            if is_mcp:
                mcp_count += 1
                prefix = "[MCP]"
            else:
                local_count += 1
                prefix = "[LOCAL]"

            print(f"{i}. {prefix} {tool_name}")
            if show_details:
                print(f"   Type: {tool_type}")
                print(f"   Description: {description[:100]}...")
                if hasattr(tool, "args_schema"):
                    try:
                        schema = tool.args_schema
                        print(f"   Schema: {schema}")
                    except Exception:
                        pass
                print()

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Local Tools: {local_count}")
        print(f"MCP Tools: {mcp_count}")
        print(f"Total: {len(tools)}")
        print()

        if enable_mcp and mcp_count == 0:
            print("⚠️  MCP enabled but no MCP tools found. Check:")
            print("   - langchain_mcp_adapters is installed")
            print("   - MCP servers are running and accessible")
            print("   - mcp_client.py configuration is correct")

    finally:
        await tm.aclose()


if __name__ == "__main__":
    print(f"MCP Enabled: {('--mcp' in sys.argv)}")
    print(f"Show Details: {('--details' in sys.argv)}\n")
    asyncio.run(main())
