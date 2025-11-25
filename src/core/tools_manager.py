from typing import List, Optional, Any
from src.tools.local_tools import SearchTool, StockPriceTool, RAGTool
import asyncio


class ToolsManager:
    """Async-first manager for local and MCP tools."""

    def __init__(self, enable_mcp: bool = True, mcp_client: Optional[Any] = None):
        self.enable_mcp = enable_mcp
        # Normalize local tools to the shapes expected by callers: the
        # SearchTool wrapper exposes a `.tool` attribute (an actual tool
        # instance), while StockPriceTool and RAGTool are BaseTool subclasses.
        # Return the underlying tool instances here so consumers can bind
        # them directly to LLMs.
        self.local_tools = [SearchTool().tool, StockPriceTool(), RAGTool()]
        self._mcp_client = mcp_client
        self._remote_mgr: Optional[Any] = None

    @classmethod
    async def create(cls, enable_mcp: bool = True, mcp_client: Optional[Any] = None) -> "ToolsManager":
        """Async factory to initialize remote MCP manager if requested."""
        inst = cls(enable_mcp=enable_mcp, mcp_client=mcp_client)
        if enable_mcp:
            # Import lazily to avoid top-level dependency on MCP adapters
            from src.tools.mcp_tool import RemoteMCPTools
            inst._remote_mgr = await RemoteMCPTools.async_init(mcp_client)
        return inst

    @classmethod
    def create_sync(cls, enable_mcp: bool = True, mcp_client: Optional[Any] = None) -> "ToolsManager":
        """Synchronous factory for callers that don't run an event loop.

        This will block the current thread while initializing the async
        RemoteMCPTools manager.
        """
        return asyncio.run(cls.create(enable_mcp=enable_mcp, mcp_client=mcp_client))

    async def get_all_tools(self) -> List[Any]:
        """Return local tools plus MCP tools (loaded asynchronously)."""
        tools = self.local_tools.copy()
        if self.enable_mcp and self._remote_mgr is not None:
            try:
                mcp_tools = await self._remote_mgr.load_tools()
                tools.extend(mcp_tools)
            except Exception as e:
                print(f"Failed to load MCP tools: {e}")
        return tools

    def get_all_tools_sync(self) -> List[Any]:
        """Synchronous convenience wrapper around `get_all_tools`.

        Useful for scripts that aren't async. This will run the event loop
        and return the combined tool list.
        """
        return asyncio.run(self.get_all_tools())

    async def aclose(self) -> None:
        """Async close/cleanup of remote manager."""
        if self._remote_mgr is not None:
            await self._remote_mgr.aclose()
            self._remote_mgr = None

    async def __aenter__(self) -> "ToolsManager":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # Optional sync convenience for callers that are not async
    def close(self) -> None:
        """Synchronous close (runs async aclose)."""
        return asyncio.run(self.aclose())
# ...existing code...