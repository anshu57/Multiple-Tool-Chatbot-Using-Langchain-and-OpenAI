import asyncio
import os
from typing import List, Annotated, Dict, Any
from fastapi import UploadFile
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.llm_provider import ChatModelProvider
from src.tools.local_tools import RAGTool, StockPriceTool, SearchTool
from src.core.vector_store import VectorStoreManager
from src.core.tools_manager import ToolsManager
from src.core.logger import get_logger


logger = get_logger(__name__)
class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.
    """
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]
    thread_id: str


class ChatbotManager:
    """
    Manages the LangGraph-based chatbot, including state, tools, and execution.
    """

    def __init__(self, checkpointer=None, tools_manager: ToolsManager = None):
        """
        Initializes the ChatbotManager.

        Args:
            checkpointer: A LangGraph checkpointer for persistence. Defaults to MemorySaver.
            tools_manager: A ToolsManager instance. If None, will use local tools only.
        """
        self.checkpointer = checkpointer
        self.tools_manager = tools_manager
        
        # In-memory storage for retrievers and metadata per thread
        self._thread_retrievers: Dict[str, VectorStoreRetriever] = {}
        self._thread_metadata: Dict[str, Dict[str, Any]] = {}

        # 1. Initialize providers and tools
        self.llm_provider = ChatModelProvider()
        
        # Get tools: use tools_manager if provided, otherwise fall back to local tools
        if tools_manager is not None:
            # Tools manager is already initialized, use its local tools
            self.tools = tools_manager.local_tools.copy()
        else:
            # Fallback to hardcoded local tools for backward compatibility
            self.tools = [RAGTool(), StockPriceTool(), SearchTool().tool]

        # 2. Bind tools to the LLM
        self.llm_with_tools = self.llm_provider.get_llm().bind_tools(self.tools)

        # 3. Define the LangGraph
        self.graph = self._build_graph()

    @classmethod
    async def create(cls, checkpointer=None, enable_mcp: bool = False):
        """
        Async factory to create ChatbotManager with full tool support (local + MCP).
        
        Args:
            checkpointer: A LangGraph checkpointer for persistence.
            enable_mcp: Whether to enable remote MCP tools.
        
        Returns:
            ChatbotManager instance with all tools initialized.
        """
        # Create tools manager with MCP support if requested
        tools_manager = await ToolsManager.create(enable_mcp=enable_mcp)
        
        # Load all tools (local + MCP if enabled)
        all_tools = await tools_manager.get_all_tools()
        
        # Create instance with the tools manager
        inst = cls(checkpointer=checkpointer, tools_manager=tools_manager)
        
        # Override tools with the full list (local + MCP)
        inst.tools = all_tools
        
        # Re-bind tools to LLM with the updated list
        inst.llm_with_tools = inst.llm_provider.get_llm().bind_tools(inst.tools)
        
        # Rebuild graph with updated tools
        inst.graph = inst._build_graph()
        
        return inst

    async def add_document_to_thread(self, thread_id: str, file: UploadFile) -> None:
        """
        Creates a retriever for a given PDF and associates it with a thread_id.
        """
        manager = VectorStoreManager()
        retriever = await manager.create_from_upload(file)
        
        self._thread_retrievers[thread_id] = retriever
        self._thread_metadata[thread_id] = {"filename": file.filename}
        logger.info(f"Associated PDF '{file.filename}' with thread_id '{thread_id}'.")

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph structure with nodes and edges.
        """
        graph = StateGraph(AgentState)

        # Define the agent node
        graph.add_node("agent", self._call_model)

        # Define the tool execution node
        graph.add_node("tools", self._call_tool)

        # Define the conditional edge
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END}
        )

        # Define the edge from tools back to the agent
        graph.add_edge("tools", "agent")

        # Set the entry point
        graph.set_entry_point("agent")

        return graph.compile(checkpointer=self.checkpointer)

    def _should_continue(self, state: AgentState) -> str:
        """
        Determines the next step: call tools or end the turn.
        """
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    async def _call_model(self, state: AgentState) -> dict:
        """
        The agent node: invokes the LLM with the current state.
        """
        # Debug: inspect the messages we're about to send to the model
        try:
            logger.debug("[_call_model] Messages about to be sent to LLM:")
            for i, m in enumerate(state["messages"]):
                t = type(m).__name__
                tool_calls = getattr(m, "tool_calls", None)
                logger.debug(f"  index={i} type={t} id={getattr(m, 'id', None)} tool_calls={tool_calls}")
        except Exception as e:
            logger.warning(f"[_call_model] Failed to stringify messages for debugging: {e}")

        response = await self.llm_with_tools.ainvoke(state["messages"])

        # Normalize the response to a message object. Some LLM wrappers
        # return wrapper objects (e.g., ChatResult) while others return
        # a Message directly. Ensure we return a Message instance so the
        # graph downstream can inspect .tool_calls reliably.
        msg = None
        # Common shapes: direct message, object with .message, object with .messages
        if isinstance(response, BaseMessage):
            msg = response
        elif hasattr(response, "message"):
            msg = response.message
        elif hasattr(response, "messages") and response.messages:
            # take the first message if it's a list-like container
            msg = response.messages[0]
        elif isinstance(response, list) and response:
            msg = response[0]
        else:
            # Fallback: wrap as HumanMessage (should not normally happen)
            msg = HumanMessage(content=str(response))

        return {"messages": [msg]}

    async def _call_tool(self, state: AgentState) -> dict:
        """
        The tool node: executes the tool calls made by the agent.
        """
        # Be defensive about the shape of the tool call object: it can be
        # a dict or an object with attributes. Normalize access.
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or last_msg
        # If tool_calls is actually the message itself, try to extract
        # tool_calls attribute again (handles some wrapper shapes).
        if not isinstance(tool_calls, list):
            tool_calls = getattr(last_msg, "tool_calls", [])

        tool_outputs = []
        for tool_call in tool_calls:
            # normalize id/name/args access
            tc_id = getattr(tool_call, "id", None) or (tool_call.get("id") if isinstance(tool_call, dict) else None)
            tc_name = getattr(tool_call, "name", None) or (tool_call.get("name") if isinstance(tool_call, dict) else None)
            tc_args = getattr(tool_call, "args", None) or (tool_call.get("args") if isinstance(tool_call, dict) else {})

            # find the tool by comparing names (defensive getattr)
            tool_to_call = next((t for t in self.tools if getattr(t, "name", None) == tc_name), None)

            logger.debug(f"[_call_tool] Handling tool_call id={tc_id} name={tc_name} args={tc_args}")
            if not tool_to_call:
                output = f"Error: Tool '{tc_name}' not found."
            else:
                # Special handling for RAGTool to provide the retriever
                if tc_name == 'rag_tool':
                    retriever = self._thread_retrievers.get(state['thread_id'])
                    if not retriever:
                        output = {"error": "No document indexed for this chat. Upload a PDF first."}
                    else:
                        # The BaseTool.ainvoke path does not forward arbitrary kwargs
                        # into the tool's `_arun` call. Call the RAGTool implementation
                        # directly so the retriever kwarg is received by `_arun`.
                        try:
                            output = await tool_to_call._arun(tc_args.get("query") if isinstance(tc_args, dict) else tc_args, retriever=retriever)
                        except Exception:
                            # Fallback to .ainvoke if direct call fails (defensive)
                            output = await tool_to_call.ainvoke(tc_args or {}, retriever=retriever)
                else:
                    output = await tool_to_call.ainvoke(tc_args or {})

            # Clean up if provided
            if tc_name == 'rag_tool' and isinstance(tc_args, dict):
                tc_args.pop('thread_id', None)

            # Build a ToolMessage that explicitly includes the tool_call_id and name
            tool_msg = ToolMessage(content=str(output), tool_call_id=tc_id, name=tc_name)
            tool_outputs.append(tool_msg)

        return {"messages": tool_outputs}

    async def aclose(self) -> None:
        """Cleanup resources, including tools manager if used."""
        if self.tools_manager is not None:
            await self.tools_manager.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def invoke(self, thread_id: str, message: str) -> str:
        """
        Main entry point to interact with the chatbot.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # The user's message
        user_message = HumanMessage(content=message)
        
        # The state to pass to the graph
        current_state = {"messages": [user_message], "thread_id": thread_id}

        # Invoke the graph to get the final state
        final_state = await self.graph.ainvoke(current_state, config=config)
        
        # The final response from the agent is the last message
        final_response = final_state["messages"][-1].content
        
        return final_response

    async def stream(self, thread_id: str, message: str):
        """
        Streams the chatbot's response for a given thread and message.
        Yields response chunks as they become available.
        """
        import json
        
        config = {
            "configurable": {"thread_id": thread_id, "checkpointer": self.checkpointer}
        }
        user_message = HumanMessage(content=message)
        current_state = {"messages": [user_message], "thread_id": thread_id}

        # Use astream_events to get token-level streaming
        async for event in self.graph.astream_events(current_state, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Yield the token as a server-sent event (SSE) formatted JSON string
                    data = json.dumps({"content": content})
                    yield f"data: {data}\n\n"