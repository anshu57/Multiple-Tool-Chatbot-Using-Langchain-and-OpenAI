"""
Streamlit Frontend for LangGraph Chatbot with MCP Tools

This provides a user-friendly web interface to interact with the chatbot API.
Supports streaming responses, conversation history, and tool information.
"""

import streamlit as st
import requests
import json
import uuid
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
import os

# Page configuration
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container layout */
    .main {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding-bottom: 0 !important;
    }
    
    body {
        background-color: #f5f5f5;
        margin: 0;
        padding: 0;
    }
    
    /* Chat messages */
    .message-user {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        margin-left: 40px;
        border-left: 4px solid #2196f3;
        word-wrap: break-word;
    }
    
    .message-assistant {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        margin-right: 40px;
        border-left: 4px solid #667eea;
        word-wrap: break-word;
    }
    
    /* Fixed input container at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, white, white);
        padding: 16px 24px;
        border-top: 1px solid #e0e0e0;
        box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    
    /* Scrollable chat area */
    .chat-area {
        flex: 1;
        overflow-y: auto;
        padding-bottom: 150px;
    }
    
    /* Hide streamlit footer */
    footer {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- Session State Management ---
def initialize_session_state():
    """Initializes the session state for chat history."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        # Create a default first chat
        first_thread_id = str(uuid.uuid4())
        st.session_state.chats[first_thread_id] = {
            "title": "New Chat",
            "messages": [],
            "thread_id": first_thread_id
        }
        st.session_state.active_thread_id = first_thread_id
initialize_session_state()

# Simple Header
st.title("🤖 LangGraph Chatbot")
st.caption("Chat with an intelligent agent powered by LangGraph and MCP tools")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("☰ Menu")
    
    # New Chat Button
    if st.button("🆕 New Chat", use_container_width=True):
        new_thread_id = str(uuid.uuid4())
        st.session_state.chats[new_thread_id] = {
            "title": "New Chat",
            "messages": [],
            "thread_id": new_thread_id
        }
        st.session_state.active_thread_id = new_thread_id
        st.rerun()
    
    st.divider()

    # Chat History
    st.subheader("📜 Chat History")
    for thread_id, chat_info in reversed(list(st.session_state.chats.items())):
        if st.button(chat_info["title"], key=f"chat_{thread_id}", use_container_width=True):
            st.session_state.active_thread_id = thread_id
            st.rerun()
    
    st.divider()
    
    st.subheader("� Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            # Send the file content directly to the backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(
                f"{API_BASE_URL}/upload-pdf/{st.session_state.active_thread_id}",
                files=files,
                timeout=30
            )
            if response.status_code == 200:
                st.success(f"✓ Uploaded: {uploaded_file.name}")
                st.info("✓ PDF associated with conversation")
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Failed to upload PDF: {response.status_code} - {error_detail}")
        except Exception as e:
            st.error(f"Error uploading PDF: {str(e)}")
    
    st.divider()
    
    # API Status
    st.subheader("Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✓ API Connected")
        else:
            st.error("✗ API Error")
    except:
        st.error("✗ Cannot connect to API")

# --- Main Chat Area ---
active_chat = st.session_state.chats[st.session_state.active_thread_id]

# Display existing messages
for msg in active_chat["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if not active_chat["messages"]:
    st.info("👋 Start a conversation by typing a message below!")

# Add spacer at bottom
st.write("")
st.write("")
st.write("")
st.write("")

def stream_response(user_input: str):
    """Stream response from API and yield content chunks."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/{st.session_state.active_thread_id}",
            params={"message": user_input.strip()},
            stream=True,
            timeout=120
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                try:
                    data = json.loads(line[6:])
                    yield data.get("content", "")
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout")
        yield ""
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {e}")
        yield ""
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        yield ""

# Fixed Input at Bottom
if user_input := st.chat_input("Type your message..."):
    # If this is the first message in a "New Chat", create a title
    if active_chat["title"] == "New Chat" and not active_chat["messages"]:
        # Generate a title from the first 5 words of the user's message
        title = " ".join(user_input.split()[:5])
        active_chat["title"] = title if title else "Chat"

    # Add user message to history and display it immediately
    active_chat["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    st.rerun() # Rerun to show the user message and update the history list

# The logic to get the bot's response should only run if the last message was from the user
if active_chat["messages"] and active_chat["messages"][-1]["role"] == "user":
    last_user_message = active_chat["messages"][-1]["content"]
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_response(last_user_message))
    
    if full_response:
        active_chat["messages"].append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# Footer
st.caption("🤖 LangGraph Chatbot v2.0 | Powered by OpenAI, LangGraph & MCP")
