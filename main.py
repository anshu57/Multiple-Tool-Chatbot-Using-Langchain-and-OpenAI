"""
FastAPI app.py - Updated to use ChatbotManager with MCP tools

This shows how to initialize the chatbot with MCP tools in a FastAPI app.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.backend.manager import ChatbotManager
from dotenv import load_dotenv

load_dotenv()


# Global chatbot instance
chatbot: ChatbotManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage chatbot lifecycle:
    - Initialize on startup
    - Clean up on shutdown
    """
    global chatbot
    print("🚀 Starting up... Initializing chatbot with MCP tools")
    try:
        chatbot = await ChatbotManager.create(enable_mcp=True)
        print(f"✓ Chatbot initialized with {len(chatbot.tools)} tools")
    except Exception as e:
        print(f"⚠️  Failed to initialize MCP tools, falling back to local tools: {e}")
        chatbot = ChatbotManager()  # Fallback to local tools
    
    yield
    
    print("🛑 Shutting down... Cleaning up chatbot")
    if chatbot:
        await chatbot.aclose()
    print("✓ Chatbot cleaned up")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="LangGraph Chatbot with MCP Tools",
    description="Agent-based chatbot with local and remote MCP tools",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return {
        "status": "healthy",
        "tools": len(chatbot.tools),
        "ready": True
    }


@app.get("/tools")
async def list_tools():
    """List all available tools."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    tools = []
    for tool in chatbot.tools:
        tool_name = getattr(tool, "name", type(tool).__name__)
        tool_desc = getattr(tool, "description", "No description")
        tool_type = type(tool).__name__
        is_mcp = "StructuredTool" in tool_type
        
        tools.append({
            "name": tool_name,
            "description": tool_desc,
            "type": tool_type,
            "is_mcp": is_mcp
        })
    
    local_count = sum(1 for t in tools if not t["is_mcp"])
    mcp_count = sum(1 for t in tools if t["is_mcp"])
    
    return {
        "total": len(tools),
        "local": local_count,
        "mcp": mcp_count,
        "tools": tools
    }


@app.get("/chat/{thread_id}")
async def chat_stream(thread_id: str, message: str):
    """
    Stream chat response for a given thread and message.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        message: User's message
    
    Returns:
        Server-Sent Event stream with response chunks
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        return StreamingResponse(
            chatbot.stream(thread_id, message.strip()),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat/{thread_id}")
async def chat_invoke(thread_id: str, message: str):
    """
    Get full chat response (non-streaming).
    
    Args:
        thread_id: Unique identifier for the conversation thread
        message: User's message
    
    Returns:
        Full response from the chatbot
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = await chatbot.invoke(thread_id, message.strip())
        return {
            "thread_id": thread_id,
            "message": message,
            "response": response,
            "tools_available": len(chatbot.tools)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upload-pdf/{thread_id}")
async def upload_pdf(thread_id: str, pdf_path: str):
    """
    Associate a PDF document with a thread for RAG.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        pdf_path: Path to the PDF file
    
    Returns:
        Confirmation of PDF association
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        await chatbot.add_document_to_thread(thread_id, pdf_path)
        return {
            "thread_id": thread_id,
            "pdf_path": pdf_path,
            "status": "success",
            "message": f"PDF associated with thread {thread_id}"
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("Starting LangGraph Chatbot API with MCP Tools")
    print("📚 Available tools include:")
    print("  - Local: duckduckgo_search, stock_price, rag_tool")
    print("  - MCP: calculator, add_expense, list_expenses, summarize")
    print()
    print("Server: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
