import streamlit as st
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging
import re

# Import core modular modules
from document_parser import MultiModalDocumentParser
from database_manager import RAGDatabaseManager
from agent_engine import agent_app  # Import our LangGraph engine

def clean_deepseek_response(text: str) -> str:
    """Removes the internal <think>...</think> monologue blocks from DeepSeek output."""
    if not text:
        return ""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_final_agent_response(final_state) -> str:
    """
    Safely iterates backward through LangGraph state messages to find the TRUE final answer,
    ignoring ToolMessages, HumanMessages, or intermediate empty AIMessages.
    """
    if not final_state or "messages" not in final_state:
        return "I couldn't process that request properly. Please try again."
        
    messages = final_state.get("messages", [])
    
    # Iterate backward through messages
    for msg in reversed(messages):
        # 1. Skip ToolMessages / SystemMessages / HumanMessages
        msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
        if msg_type in ["tool", "human", "system"]:
            continue

        # 2. Extract content
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
            
        if isinstance(content, str):
            cleaned = clean_deepseek_response(content)
            # Ensure it's not empty and not just tool call structure
            if cleaned and not cleaned.startswith("{") and len(cleaned) > 2:
                return cleaned

    return "The agent executed tools but did not return a final written summary."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RAGAi")

load_dotenv()

st.set_page_config(page_title="LangGraph Multimodal AI Agent Hub", page_icon="🤖", layout="wide")

STATIC_ASSET_DIR = "./processed_data"
os.makedirs(STATIC_ASSET_DIR, exist_ok=True)
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@st.cache_resource
def initialize_rag_services():
    parser = MultiModalDocumentParser(base_output_dir=STATIC_ASSET_DIR, batch_size=3)
    db_manager = RAGDatabaseManager(db_path=os.path.join(STATIC_ASSET_DIR, "rag_storage.db"))
    return parser, db_manager

parser_engine, db_engine = initialize_rag_services()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "inspected_nodes" not in st.session_state:
    st.session_state.inspected_nodes = []

# Styling
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    .stCodeBlock { background-color: #161b26 !important; }
    div[data-testid="stExpander"] { background-color: #161b26 !important; border: 1px solid #2d3748 !important; }
    div[data-testid="stChatMessage"] { background-color: #111827 !important; border-radius: 8px; margin-bottom: 6px; padding: 10px; }
    div[data-testid="stStatusWidget"] { background-color: #161b26 !important; border: 1px solid #10b981 !important; }
    h3 { padding-top: 10px !important; padding-bottom: 10px !important; }
    hr { margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 LangGraph Multimodal AI Agent Hub")
st.divider()

# =====================================================================
# MAIN VIEWPORT LAYOUT (LEFT: 80% | RIGHT SIDEBAR: 20%)
# =====================================================================
main_left, main_right = st.columns([0.8, 0.2], gap="small")

# ---------------------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------------------
with main_left:
    chat_col, inspector_col = st.columns([1, 1], gap="small")

    with chat_col:
        st.subheader("💬 Active Conversation Space")
        chat_container = st.container(height=580, border=True)
        with chat_container:
            for msg in st.session_state.chat_history:
                # Strictly only render messages that have non-empty valid content
                if msg.get("content") and msg["content"].strip():
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

    with inspector_col:
        st.subheader("🔍 Context & Asset Inspector")
        inspector_container = st.container(height=580, border=True)
        with inspector_container:
            if st.session_state.inspected_nodes:
                st.caption("Active Visual & Data Context retrieved by Agent Tool:")
                for index, match in enumerate(st.session_state.inspected_nodes):
                    with st.container(border=True):
                        st.markdown(f"📄 **Source:** `{match.get('filename', 'Unknown')}`")
                        
                        if match.get("text"):
                            with st.expander("🔬 View Text / Schema Context", expanded=True):
                                st.write(match["text"])
                                
                        if match.get("table_path") and os.path.exists(match["table_path"]):
                            st.markdown("📊 **Extracted Structural Table Data:**")
                            try:
                                df_display = pd.read_csv(match["table_path"])
                                st.dataframe(df_display, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Error reading asset table: {e}")
                        
                        if match.get("image_path") and os.path.exists(match["image_path"]):
                            st.markdown("🖼️ **Extracted Diagram / Figure Asset:**")
                            st.image(match["image_path"], use_container_width=True)
            else:
                st.info("💡 Any extracted tables, schemas, or diagram layouts selected by the agent tools will stack right here.")

    st.divider()

    # Chat Input Section
    st.subheader("💬 Ask to Agent")
    user_query = st.chat_input("Message your local knowledge agent...")

    if user_query:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)

            status_placeholder = st.empty()
            
            with status_placeholder.container():
                with st.status("🧠 Agent Evaluating & Planning...", expanded=True) as status:
                    st.write("Initializing state nodes...")

                    # Send ONLY clean, non-empty conversation turns to LangGraph
                    langgraph_messages = []
                    for m in st.session_state.chat_history:
                        if m.get("content") and m["content"].strip():
                            role_tag = "user" if m["role"] == "user" else "assistant"
                            langgraph_messages.append((role_tag, m["content"]))

                    node_holder = []
                    agent_config = {
                        "configurable": {
                            "db_manager": db_engine,
                            "retrieval_limit": st.session_state.get("retrieval_k_slider", 3),
                            "ollama_base_url": ollama_base_url,
                            "temperature": st.session_state.get("temp_slider", 0.1),
                            "top_k": st.session_state.get("top_k_slider", 40),
                            "shared_node_container": node_holder
                        }
                    }

                    try:
                        stream_generator = agent_app.stream(
                            {"messages": langgraph_messages}, 
                            config=agent_config,
                            stream_mode="values"
                        )
                        
                        final_state = None
                        for market_chunk in stream_generator:
                            final_state = market_chunk
                            if "messages" in market_chunk and market_chunk["messages"]:
                                last_msg = market_chunk["messages"][-1]
                                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                    for tc in last_msg.tool_calls:
                                        st.write(f"⚙️ **Invoked Tool**: `{tc.get('name')}`")
                                        
                            if node_holder:
                                st.session_state.inspected_nodes = node_holder

                        status.update(label="✅ Execution Path Complete", state="complete")
                    except Exception as e:
                        logger.error(f"Error during agent execution: {e}")
                        status.update(label="❌ Execution Failed", state="error")
                        final_state = None

            # Extract final response safely
            final_answer = extract_final_agent_response(final_state)

            # Prevent saving identical consecutive questions or empty strings as assistant answers
            if final_answer and final_answer != user_query:
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
            else:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "I couldn't synthesize a new response for this step. Please rephrase or check the database index."
                })

            status_placeholder.empty()
            st.rerun()

    st.divider()

    # Ingestion Vault Section
    st.subheader("📂 Ingestion Vault")
    uploaded_files = st.file_uploader(
        "Upload files:", 
        type=["pdf", "docx", "xlsx", "csv", "md", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Build Knowledge Base Data", use_container_width=True):
        active_chunk_size = st.session_state.get("chunk_size_slider", 1000)
        active_chunk_overlap = st.session_state.get("chunk_overlap_slider", 200)

        for uploaded_file in uploaded_files:
            with st.spinner(f"Parsing and processing: {uploaded_file.name}..."):
                try:
                    chunks = parser_engine.parse_file(
                        uploaded_file.getvalue(), 
                        uploaded_file.name, 
                        chunk_size=active_chunk_size, 
                        chunk_overlap=active_chunk_overlap
                    )
                    if chunks:
                        db_engine.insert_document_chunks(chunks, uploaded_file.name)
                        st.success(f"Successfully processed & indexed: {uploaded_file.name}")
                except Exception as ex:
                    st.error(f"Failed parsing {uploaded_file.name}: {str(ex)}")

# ---------------------------------------------------------------------
# RIGHT COLUMN: AGENT SETTINGS
# ---------------------------------------------------------------------
with main_right:
    st.subheader("⚙️ Agent Settings")
    
    settings_container = st.container(height=840, border=True)
    with settings_container:
        st.caption("Framework: **LangGraph State Machine**")
        st.divider()
        
        st.markdown("##### ✂️ Chunking Hyperparameters")
        st.slider("Max Chunk Size", min_value=200, max_value=3000, value=1000, step=50, key="chunk_size_slider")
        st.slider("Chunk Overlap Block", min_value=0, max_value=500, value=200, step=10, key="chunk_overlap_slider")
        
        st.divider()
        st.markdown("##### 🎛️ Agent Parameters")
        st.slider("Database Chunks (Retrieval K)", min_value=1, max_value=10, value=3, step=1, key="retrieval_k_slider")
        st.slider("Generation Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temp_slider")
        st.slider("Generation Top-K Window", min_value=1, max_value=100, value=40, step=1, key="top_k_slider")

        st.divider()
        st.write("") 
        if st.button("🗑️ Wipe Knowledge Base Data", use_container_width=True):
            db_engine.wipe_all_data()
            tabular_db = Path("./processed_data/dynamic_tabular_data.db")
            if tabular_db.exists():
                os.remove(tabular_db)
            st.session_state.inspected_nodes = []
            st.success("Internal state flushed!")
            
        if st.button("💬 Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.inspected_nodes = []
            st.rerun()