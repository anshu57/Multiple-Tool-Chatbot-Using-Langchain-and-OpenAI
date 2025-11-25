"""
Streamlit Frontend for LangGraph Chatbot with MCP Tools

This provides a user-friendly web interface to interact with the chatbot API.
Supports streaming responses, conversation history, and tool information.
"""

import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import Generator
import time
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

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

# Simple Header
st.title("🤖 LangGraph Chatbot")
st.caption("Chat with an intelligent agent powered by LangGraph and MCP tools")

st.divider()

# Sidebar with PDF Upload and New Chat
with st.sidebar:
    st.header("☰ Menu")
    
    # New Chat Button at top
    if st.button("� New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.subheader("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file
        temp_path = f"temp_files/{uploaded_file.name}"
        os.makedirs("temp_files", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        # Associate with thread
        try:
            response = requests.post(
                f"{API_BASE_URL}/upload-pdf/{st.session_state.thread_id}",
                params={"pdf_path": temp_path},
                timeout=30
            )
            if response.status_code == 200:
                st.info("✓ PDF associated with conversation")
            else:
                st.error(f"Failed to upload PDF: {response.status_code}")
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

# Main Chat Area
if st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="message-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-assistant"><b>Bot:</b> {msg["content"]}</div>', unsafe_allow_html=True)
else:
    st.info("👋 Start a conversation by typing a message below!")

# Add spacer at bottom
st.write("")
st.write("")
st.write("")
st.write("")

# Fixed Input at Bottom using columns
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Create form for Enter key support
with st.form(key="message_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Message:",
            placeholder="Type your message and press Enter...",
            label_visibility="collapsed",
            key="message_input"
        )
    
    with col2:
        submit = st.form_submit_button("Send", use_container_width=True)
    
    # Handle message submission
    if submit and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream response from API
        try:
            with st.spinner("🤔 Thinking..."):
                response = requests.get(
                    f"{API_BASE_URL}/chat/{st.session_state.thread_id}",
                    params={"message": user_input.strip()},
                    stream=True,
                    timeout=120
                )
                
                if response.status_code == 200:
                    # Collect streamed response
                    full_response = ""
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                # Parse Server-Sent Event format
                                if line.startswith(b"data: "):
                                    data = json.loads(line[6:])
                                    chunk = data.get("content", "")
                                    full_response += chunk
                            except json.JSONDecodeError:
                                continue
                    
                    # Add assistant response to history
                    if full_response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        st.warning("⚠️ No response from server")
                else:
                    st.error(f"⚠️ API Error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.caption("🤖 LangGraph Chatbot v2.0 | Powered by OpenAI, LangGraph & MCP")
