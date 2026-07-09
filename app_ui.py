import streamlit as st
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

# Import core modular modules
from document_parser import MultiModalDocumentParser
from database_manager import RAGDatabaseManager
from agent_engine import agent_app  # Import our newly constructed LangGraph machine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAGAi")

load_dotenv()
st.set_page_config(page_title="Multi-Modal Knowledge Vault UI", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph Multimodal AI Agent Hub")

STATIC_ASSET_DIR = "./processed_data"
os.makedirs(STATIC_ASSET_DIR, exist_ok=True)
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@st.cache_resource
def initialize_rag_services():
    parser = MultiModalDocumentParser(base_output_dir=STATIC_ASSET_DIR, batch_size=3)
    db_manager = RAGDatabaseManager(db_path=os.path.join(STATIC_ASSET_DIR, "rag_storage.db"))
    return parser, db_manager

parser_engine, db_engine = initialize_rag_services()

# Keep persistent historical state intact across continuous slider re-renders
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "inspected_nodes" not in st.session_state:
    st.session_state.inspected_nodes = []
st.markdown(
    """
    <style>
    /* Force completely dark style properties onto text containers */
    .stCodeBlock {
        background-color: #161b26 !important;
    }
    div[data-testid="stExpander"] {
        background-color: #161b26 !important;
        border: 1px solid #2d3748 !important;
    }
    div[data-testid="stChatMessage"] {
        background-color: #111827 !important;
        border-radius: 8px;
        margin-bottom: 10px;
        padding: 15px;
    }
    /* Subtle neon glow for your agent's reasoning loop card */
    div[data-testid="stStatusWidget"] {
        background-color: #161b26 !important;
        border: 1px solid #10b981 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.caption("Orchestrator framework: **LangGraph State Machine**")
    
    st.markdown("---")
    st.subheader("✂️ Chunking Hyperparameters")
    chunk_size = st.slider("Max Chunk Size", min_value=200, max_value=3000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap Block", min_value=0, max_value=500, value=200, step=20)
    
    st.markdown("---")
    st.subheader("🎛️ Agent Model Parameters")
    retrieval_top_k = st.slider("Database Chunks (Retrieval K)", min_value=1, max_value=10, value=3, step=1)
    generation_temperature = st.slider("Generation Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    generation_top_k = st.slider("Generation Top-K Window", min_value=1, max_value=100, value=40, step=1)

    st.markdown("---")
    if st.button("Wipe Index Matrix", use_container_width=True):
        db_engine.wipe_all_data()
        st.success("Internal state matrices flushed!")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.inspected_nodes = []
        st.rerun()

# --- INGESTION VAULT PANEL ---
st.subheader("📂 Ingestion Vault")
uploaded_files = st.file_uploader(
    "Upload files:", type=["pdf", "docx", "xlsx", "csv", "md", "txt"], accept_multiple_files=True
)

if uploaded_files and st.button("🚀 Build Knowledge Base Index", use_container_width=True):
    for uploaded_file in uploaded_files:
        with st.spinner(f"Parsing layouts: {uploaded_file.name}..."):
            try:
                chunks = parser_engine.parse_file(
                    uploaded_file.getvalue(), uploaded_file.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                if chunks:
                    db_engine.insert_document_chunks(chunks, uploaded_file.name)
                    st.success(f"Indexed: {uploaded_file.name} successfully!")
            except Exception as ex:
                st.error(f"Failed parsing {uploaded_file.name}: {str(ex)}")

st.markdown("---")

# --- TWO-COLUMN WORKSPACE FRAME (Left=Chat, Right=Asset Inspector) ---
chat_canvas, asset_inspector = st.columns([3, 2], gap="large")

with chat_canvas:
    st.subheader("💬 Active Conversation Space")
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_query := st.chat_input("Message your local knowledge agent..."):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Prepare the conversation payload state
        langgraph_messages = []
        for m in st.session_state.chat_history:
            role_tag = "user" if m["role"] == "user" else "assistant"
            langgraph_messages.append((role_tag, m["content"]))
            
        # Configure dynamic graph parameter routing dictionary
        agent_config = {
            "configurable": {
                "db_manager": db_engine,
                "retrieval_limit": retrieval_top_k,
                "ollama_base_url": ollama_base_url,
                "temperature": generation_temperature,
                "top_k": generation_top_k
            }
        }

        with chat_container:
            with st.chat_message("assistant"):
                # Initialize our status container for tool tracking
                with st.status("🧠 Agent Evaluating & Planning...", expanded=True) as status:
                    st.write("Initializing state nodes...")
                    
                    # 1. Use .stream() instead of .invoke() to get fine-grained graph updates
                    # we specify stream_mode="values" to track state content updates continuously
                    stream_generator = agent_app.stream(
                        {"messages": langgraph_messages}, 
                        config=agent_config,
                        stream_mode="values"
                    )
                    
                    # We will collect the execution payloads as the graph processes them
                    final_state = None
                    for market_chunk in stream_generator:
                        final_state = market_chunk
                        # If a tool call has been packed into the latest message, show it live
                        if "messages" in market_chunk and market_chunk["messages"]:
                            last_msg = market_chunk["messages"][-1]
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                st.write(f"⚙️ **Invoked Tool**: `search_knowledge_base` with query: *\"{last_msg.tool_calls[0]['args']['query']}\"*")
                    
                    status.update(label="✅ Execution Path Complete", state="complete")
                
                # 2. Render real-time token streaming onto the chat canvas
                # st.write_stream accepts any generator yielding strings and handles the UI updates perfectly
                response_placeholder = st.empty()
                
                def token_streamer():
                    # Extract messages from the terminal graph node state
                    messages_list = final_state.get("messages", []) if final_state else []
                    if messages_list:
                        last_message = messages_list[-1]
                        # Stream the contents if the object has stream capabilities
                        if hasattr(last_message, "content"):
                            # Split into words/tokens mock chunk or yield clean chunks
                            yield last_message.content.strip()
                        else:
                            yield str(last_message).strip()

                # Stream the finalized answer instantly with typewriter effects
                final_response = response_placeholder.write_stream(token_streamer())
                
                # Append the finalized string payload back to historical view storage
                st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                
                # Fetch recent nodes directly from our live lookup to update asset inspector panel instantly
                st.session_state.inspected_nodes = db_engine.search_bm25(user_query, limit=retrieval_top_k)
                st.rerun()

# --- RIGHT HAND SIDEBAR ASSET INSPECTOR ---
with asset_inspector:
    st.subheader("🔍 Context & Asset Inspector")
    if st.session_state.inspected_nodes:
        st.caption("Active Visual Evidence mapped to your text context by the agent tool:")
        for index, match in enumerate(st.session_state.inspected_nodes):
            with st.container(border=True):
                st.markdown(f"📄 **Hit #{index + 1}** | `{match['filename']}`")
                with st.expander("🔬 View Text Context"):
                    st.write(match["text"])
                    
                if match.get("table_path") and os.path.exists(match["table_path"]):
                    try:
                        df = pd.read_csv(match["table_path"])
                        st.markdown("📊 **Associated Tabular Data Matrix:**")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    except Exception as t_err:
                        st.caption(f"Table reference present but unreadable: {t_err}")
                        
                if match.get("image_path") and os.path.exists(match["image_path"]):
                    st.markdown("🖼️ **Mapped Diagram/Crop Asset:**")
                    st.image(match["image_path"], use_container_width=True)
    else:
        st.info("💡 Any extracted tables or diagram layouts selected by the LangGraph agent tool will stack right here.")