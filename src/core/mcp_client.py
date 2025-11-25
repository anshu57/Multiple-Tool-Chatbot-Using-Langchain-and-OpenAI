import asyncio
from typing import List, Dict, Any

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MultiServerMCPClient = None


class MCPClient:
    
    def __init__(self):
        """Initialize the MCP client with server configuration."""
        if not HAS_MCP:
            print("Warning: langchain_mcp_adapters not installed. MCP tools will not be available.")
            self.client = None
            return
        
        try:
            self.client = MultiServerMCPClient(
        {
            "arith": {
                "transport": "stdio",
                "command": "python3",
                "args": ["/Users/anshugangwar/Desktop/mcp-math-server/main.py"]
            },
            "expense": {
                "transport": "streamable_http",
                "url": "https://grateful-olive-takin.fastmcp.app/mcp"
            }
        }
    )
        except Exception as e:
            print(f"Warning: Failed to initialize MCP client: {e}")
            self.client = None
    
    async def get_tools(self) -> List[Any]:
        """Get all available tools from MCP servers."""
        if self.client is None:
            return []
        try:
            tools = await self.client.get_tools()
            return tools
        except Exception as e:
            print(f"Error getting tools: {e}")
            return []
    
    async def run_tool(self, tool, args: Dict[str, Any]) -> str:
        """Run a tool with given arguments."""
        if self.client is None:
            return "Error: MCP client not initialized"
        try:
            result = await tool.ainvoke(args)
            return str(result)
        except Exception as e:
            return f"Error: {e}"