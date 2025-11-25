# ...existing code...
import asyncio
from typing import List, Any
from langchain_core.tools import StructuredTool
from src.core.mcp_client import MCPClient

class RemoteMCPTools:
    """Wrapper for MCP tools as LangChain StructuredTools (async-friendly)."""
    
    def __init__(self, mcp_client: MCPClient | None = None):
        """Initialize with an MCP client (sync constructor)."""
        self.mcp_client = mcp_client or MCPClient()
    
    @classmethod
    async def async_init(cls, mcp_client: MCPClient | None = None) -> "RemoteMCPTools":
        """
        Async initializer that will try to create/connect an MCPClient asynchronously
        if the MCPClient exposes an async factory (create/connect). Falls back to sync init.
        Usage:
            remote = await RemoteMCPTools.async_init()
        """
        if mcp_client is None:
            # Prefer async factory if available
            if hasattr(MCPClient, "create") and asyncio.iscoroutinefunction(MCPClient.create):
                mcp_client = await MCPClient.create()
            elif hasattr(MCPClient, "connect") and asyncio.iscoroutinefunction(MCPClient.connect):
                mcp_client = await MCPClient.connect()
            else:
                mcp_client = MCPClient()
        return cls(mcp_client)
    
    def _make_langchain_tool(self, mcp_tool) -> StructuredTool:
        """Convert MCP tool to LangChain StructuredTool.

        Note: bind mcp_tool into the coroutine default arg to avoid late-binding.
        Makes query parameter optional since some tools don't need it.
        """
        tool_name = mcp_tool.name
        tool_description = mcp_tool.description or "MCP tool"
        
        async def run_tool(query: str = "", *, _mcp_tool = mcp_tool) -> Any:
            """Run the MCP tool asynchronously.
            
            Args:
                query: Optional query parameter (defaults to empty string if not provided).
            """
            try:
                # Build args: include query if provided, otherwise pass empty args
                args = {"query": query} if query else {}
                # assume MCPClient.run_tool is async; await it
                result = await self.mcp_client.run_tool(_mcp_tool, args)
                return result
            except Exception as e:
                return f"Error: {e}"
        
        return StructuredTool.from_function(
            coroutine=run_tool,
            name=tool_name,
            description=tool_description
        )
    
    async def load_tools(self) -> List[StructuredTool]:
        """Load all MCP tools and convert them to LangChain StructuredTools (async)."""
        try:
            # Get raw MCP tools (awaiting async get_tools)
            mcp_tools = await self.mcp_client.get_tools()
            
            # Convert to LangChain tools
            langchain_tools: List[StructuredTool] = []
            for mcp_tool in mcp_tools:
                lc_tool = self._make_langchain_tool(mcp_tool)
                langchain_tools.append(lc_tool)
            return langchain_tools
     
        except Exception as e:
            print(f"Failed to load tools: {e}")
            return []

    def load_tools_sync(self) -> List[StructuredTool]:
        """Synchronous helper to load tools from non-async code (runs event loop)."""
        return asyncio.run(self.load_tools())

    async def aclose(self) -> None:
        """Attempt to close underlying MCP client (async if supported)."""
        if hasattr(self.mcp_client, "aclose") and asyncio.iscoroutinefunction(self.mcp_client.aclose):
            await self.mcp_client.aclose()
        elif hasattr(self.mcp_client, "close"):
            maybe_close = getattr(self.mcp_client, "close")
            if asyncio.iscoroutinefunction(maybe_close):
                await maybe_close()
            else:
                maybe_close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()
# ...existing code...