# ...existing code...
import asyncio
from typing import List, Any
from langchain_core.tools import StructuredTool
from src.core.mcp_client import MCPClient
from src.core.logger import get_logger

logger = get_logger(__name__)
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
    
    async def load_tools(self) -> List[StructuredTool]:
        """Load all MCP tools as LangChain StructuredTools (async).
        
        MCP tools from langchain_mcp_adapters are already StructuredTools
        with proper schemas, so we return them directly.
        """
        try:
            # Get MCP tools - they're already LangChain StructuredTools with proper schemas
            mcp_tools = await self.mcp_client.get_tools()
            return mcp_tools
     
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
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