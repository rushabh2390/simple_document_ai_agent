import streamlit as st
import os
import pandas as pd
from pathlib import Path
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import logging
# Hook directly into your modular back-end core files
from document_parser import MultiModalDocumentParser
from database_manager import RAGDatabaseManager
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAGAi")

load_dotenv()
st.set_page_config(page_title="Multi-Modal Knowledge Vault UI",
                   page_icon="🖼️", layout="wide")
st.title("🖼️ Docling + SQLite Multimodal RAG Hub")

# Global Configuration Data Paths
STATIC_ASSET_DIR = "./processed_data"
os.makedirs(STATIC_ASSET_DIR, exist_ok=True)
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Cache services across browser hot-reloads so they stay stable


@st.cache_resource
def initialize_rag_services():
    parser = MultiModalDocumentParser(
        base_output_dir=STATIC_ASSET_DIR, batch_size=3)
    db_manager = RAGDatabaseManager(
        db_path=os.path.join(STATIC_ASSET_DIR, "rag_storage.db"))
    return parser, db_manager


parser_engine, db_engine = initialize_rag_services()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ App Settings")
    st.caption("Active Configuration: **(SQLite)**")
    st.caption("Target Model: `llama3.2`")

    st.markdown("---")
    st.subheader("✂️ Chunking Hyperparameters")
    # Custom sliders to tune document chunk sizes dynamically
    chunk_size = st.slider("Max Chunk Size (Characters)",
                           min_value=200, max_value=3000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap Block",
                              min_value=0, max_value=500, value=200, step=20)

    if chunk_overlap >= chunk_size:
        st.error("⚠️ Overlap cannot be greater than or equal to total Chunk Size.")

    st.markdown("---")
    if st.button("🧹 Wipe Index Matrix", use_container_width=True):
        db_engine.wipe_all_data()
        st.success("Internal database state registers flushed!")
        # st.rerun()

# --- INGESTION VIEW PANEL ---
st.subheader("📂 Ingestion Vault")
uploaded_files = st.file_uploader(
    "Upload multi-format reports (.pdf, .docx, .xlsx, .csv, .md, .txt):",
    type=["pdf", "docx", "xlsx", "csv", "md", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🚀 Build Knowledge Base Index", use_container_width=True):
    for uploaded_file in uploaded_files:
        with st.spinner(f"Ingesting & extracting layouts from: {uploaded_file.name}..."):
            try:
                file_bytes = uploaded_file.getvalue()
                # Passing custom slider parameters directly downstream to your parser engine
                chunks = parser_engine.parse_file(
                    file_bytes,
                    uploaded_file.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                if chunks:
                    db_engine.insert_document_chunks(
                        chunks, uploaded_file.name)
                    st.success(f"Indexed: {uploaded_file.name} successfully!")
                else:
                    st.error(
                        f"No valid text paths found in {uploaded_file.name}.")
            except Exception as ex:
                st.error(f"Failed parsing {uploaded_file.name}: {str(ex)}")

st.markdown("---")

# --- QUERY & PRESENTATION SCREEN ---
st.subheader("💬 Ask Your Documents")
user_query = st.text_input(
    "Enter query:", placeholder="e.g., Explain tensor with example from the book")

if user_query:
    with st.spinner("Scanning cross-linked database matrices"):
        # 1. Retrieve text nodes using the updated synchronous BM25 engine
        matched_nodes = db_engine.search_bm25(user_query, limit=3)

        if not matched_nodes:
            st.warning("No matching context found.")
        else:
            logger.info("### 🔍 System Debug: Content passing to LLM")
            logger.info(f"{matched_nodes}")
            # 2. Reconstruct context cleanly without special tokens
            context_str = "\n\n".join([f"[Source: {m['filename']}]\n{m['text']}" for m in matched_nodes])
            # Use explicit LangChain Message formats to guarantee structural delivery

            messages = [
                SystemMessage(content=(
                    "You are an expert technical assistant. Answer the user's query by leveraging the provided document context below. "
                    "Provide thorough explanations, comprehensive details, and fully fleshed-out code implementations or text summaries where requested.\n\n"
                    f"--- DOCUMENT CONTEXT ---\n{context_str}"
                )),
                HumanMessage(content=user_query)
            ]

            # 3. Request inference natively via ChatOllama
            try:
                llm = ChatOllama(
                    base_url=ollama_base_url, 
                    model="llama3.2", 
                    temperature=0.3,       # Slightly increased from 0.1 to let it write code fluidly
                    num_ctx=8192,          # Large reading context window 
                    num_predict=2048       # Forces Ollama to allow up to 2048 tokens of output response text/code
                )
                response = llm.invoke(messages)
                answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            except Exception as err:
                answer = f"[Ollama Connection Error]: {err}"
                # Pass the structured messages array instead of the raw string
                response = llm.invoke(messages)
                answer = response.content.strip() if hasattr(
                    response, 'content') else str(response).strip()
            except Exception as err:
                answer = f"[Ollama Connection Error]: Make sure Ollama is open. Details: {err}"

            # Render Response Panel
            st.markdown("### 🎯 System Synthesis Answer")
            st.info(answer)

            st.markdown("### 📄 Extracted Structural Evidence")
            for index, match in enumerate(matched_nodes):
                # Safe checking for scores since returns flat metrics
                score_val = match.get("score", 0.0)
                with st.expander(f"Match #{index + 1} | File: {match['filename']} (Score Weight: {score_val:.4f})", expanded=True):
                    col_txt, col_media = st.columns([3, 2])

                    with col_txt:
                        st.markdown("**Text Fragment Context:**")
                        st.write(match["text"])
                        st.caption(
                            f"Chunk Identification ID: `{match['chunk_id']}`")

                    with col_media:
                        if match.get("table_path") and os.path.exists(match["table_path"]):
                            try:
                                df = pd.read_csv(match["table_path"])
                                st.markdown("**📊 Accompanying Data Matrix:**")
                                st.dataframe(df, use_container_width=True)
                            except Exception as table_err:
                                st.caption(
                                    f"Table context present but unreadable: {table_err}")

                        if match.get("image_path") and os.path.exists(match["image_path"]):
                            st.markdown(
                                "**🖼️ Mapped Diagram Crop Reference:**")
                            st.image(match["image_path"],
                                     use_container_width=True)

                        if not match.get("table_path") and not match.get("image_path"):
                            st.write(
                                "💡 *No supplementary visual assets mapped to this section context.*")
