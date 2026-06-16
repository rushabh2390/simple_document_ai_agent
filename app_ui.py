import os
import re
import io
import zipfile
import streamlit as st

# Docling core and pipeline options for visual element extraction
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling.datamodel.base_models import DocumentStream, InputFormat

# Lexical Search & LLM Interfaces
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEndpoint

# --- STAGE 1: DASHBOARD LAYOUT CONFIGURATION ---
st.set_page_config(
    page_title="Visual Docling RAG Vault",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Docling + BM25 Multimodal Text/Image RAG Hub")
st.markdown("Upload documents, extract structural layouts, isolate embedded diagrams, and query them seamlessly using a text index.")

# --- STAGE 2: INITIALIZING GLOBAL SESSION STATES ---
# This prevents our index matrices from getting wiped when the app updates or reruns
if "chunks_pool" not in st.session_state:
    st.session_state.chunks_pool = []  # Schema: [{"text":..., "source":..., "is_image":..., "image_bytes":...}]
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

# Local temporary output folder setup for image persistence (if needed for debugging)
STATIC_IMAGE_DIR = "./static_images"
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

# --- STAGE 3: SIDEBAR CONTROL MATRIX ---
with st.sidebar:
    st.header("⚙️ App Settings")
    
    # Masked Hugging Face Authentication Token Input
    hf_token = st.text_input("Hugging Face Access Token", type="password")
    
    st.markdown("---")
    st.subheader("📝 Text Chunking Specs")
    
    custom_chunk_size = st.slider(
        "Max Target Tokens Per Chunk", 
        min_value=100, 
        max_value=1000, 
        value=400, 
        step=50
    )
    
    st.markdown("---")
    if st.button("🧹 Wipe Index Matrix", use_container_width=True):
        st.session_state.chunks_pool = []
        st.session_state.bm25_index = None
        st.success("Internal state registries flushed!")
        st.rerun()

# --- STAGE 4: PIPELINE EXTRACTION LOGIC ---
def tokenize_text(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()

def process_and_index_docling_result(conversion_result, source_name: str, chunk_size: int):
    """
    Extracts structural markdown segments and maps embedded visual blocks 
    directly to the state pool.
    """
    doc = conversion_result.document
    
    # 1. Handle Structural Text Chunks
    markdown_text = doc.export_to_markdown()
    chunker = HybridChunker(max_tokens=chunk_size)
    doc_chunks = chunker.chunk(markdown_text)
    
    for chunk in doc_chunks:
        st.session_state.chunks_pool.append({
            "text": chunk.text,
            "source": source_name,
            "is_image": False,
            "image_bytes": None
        })
        
    # 2. Iterate through layouts to find charts, tables, and graphic models
    for element, level in doc.iterate_items():
        if element.label in ["Pictures", "Figure", "Table"]:
            if hasattr(element, "image") and element.image is not None:
                # Isolate caption string data
                caption_text = getattr(element, "caption", "Data visualization chart or corporate diagram.")
                if hasattr(caption_text, "text"):
                    caption_text = caption_text.text
                    
                # Create the text reference string for BM25 indexing
                image_metadata_anchor = f"IMAGE_REFERENCE: {element.id} | Caption/Content: {caption_text}"
                
                # Append raw visual binary directly inside the state cache
                st.session_state.chunks_pool.append({
                    "text": image_metadata_anchor,
                    "source": source_name,
                    "is_image": True,
                    "image_bytes": element.image.bytes
                })

    # 3. Compile or Hot-Reload the Global BM25 Match Map
    if st.session_state.chunks_pool:
        corpus_tokens = [tokenize_text(item["text"]) for item in st.session_state.chunks_pool]
        st.session_state.bm25_index = BM25Okapi(corpus_tokens)


# --- STAGE 5: FILE UPLOADER COMPONENT ---
st.subheader("📂 Ingestion Vault")
uploaded_files = st.file_uploader(
    "Upload multi-format reports (.pdf, .docx, .xlsx, .csv, .md, .txt) or a single compressed .zip file:",
    type=["pdf", "docx", "xlsx", "csv", "md", "txt", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Docling is running advanced visual processing loops..."):
        # Configure Docling to extract images cleanly out of the uploads
        pdf_options = PdfPipelineOptions()
        pdf_options.images_scale = 1.0  # Tells the engine to extract full-res shapes
        pipeline_options = PipelineOptions()
        converter = DocumentConverter(
            format_options={InputFormat.PDF: {"pipeline_options": pipeline_options}}
        )
        
        supported_formats = (".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".md", ".txt")
        new_files_processed = 0

        for file in uploaded_files:
            file_ext = os.path.splitext(file.name)[-1].lower()
            file_bytes = file.read()

            # CASE A: Processing Compressed ZIP files
            if file_ext == ".zip":
                try:
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
                        for arch_name in archive.namelist():
                            if arch_name.startswith("__MACOSX") or os.path.basename(arch_name) == "":
                                continue
                            
                            arch_ext = os.path.splitext(arch_name)[-1].lower()
                            if arch_ext in supported_formats:
                                with archive.open(arch_name) as inner_file:
                                    inner_bytes = inner_file.read()
                                    res = converter.convert(io.BytesIO(inner_bytes))
                                    process_and_index_docling_result(res, os.path.basename(arch_name), custom_chunk_size)
                                    new_files_processed += 1
                except Exception as e:
                    st.error(f"Failed to uncompress ZIP payload '{file.name}': {e}")

            # CASE B: Standard Individual Documents
            elif file_ext in supported_formats:
                try:
                    res = converter.convert(io.BytesIO(file_bytes))
                    process_and_index_docling_result(res, file.name, custom_chunk_size)
                    new_files_processed += 1
                except Exception as e:
                    st.error(f"Failed parsing document structural details for '{file.name}': {e}")
                    
        if new_files_processed > 0:
            st.success(f"Successfully tracking {new_files_processed} new files! Current active chunk count: {len(st.session_state.chunks_pool)}")

st.markdown("---")

# --- STAGE 6: GENERATIVE RETRIEVAL (QA LAYER) ---
st.subheader("💬 Ask Your Documents")
user_query = st.text_input("Enter query:", placeholder="e.g., Show me the revenue charts or sales summary code.")

if user_query:
    if not st.session_state.bm25_index:
        st.warning("⚠️ Access Denied: The text index is currently empty. Please drop documentation files first.")
    elif not hf_token:
        st.warning("🔑 Authentication Error: Paste your Hugging Face Access Token in the left sidebar menu configuration block.")
    else:
        with st.spinner("Scanning exact lexical matches and reaching out to Llama-3 cloud inference servers..."):
            # 1. Evaluate user search terms against the in-memory BM25 matrix
            tokenized_q = tokenize_text(user_query)
            matched_nodes = st.session_state.bm25_index.get_top_n(tokenized_q, st.session_state.chunks_pool, n=3)
            
            if not matched_nodes:
                st.error("No text segments within your active vault match those search tokens.")
            else:
                # 2. Build explicit context layout block
                context_str = "\n\n".join([f"[Source: {m['source']}]\n{m['text']}" for m in matched_nodes])
                
                # 3. Inject into custom Llama-3 instruct prompt frame
                llama_prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    f"Answer the query utilizing the factual text fragments provided below. "
                    f"If the context contains an IMAGE_REFERENCE block, mention that you have found and display corresponding graphic data diagrams cleanly in your textual explanation.\n\n"
                    f"--- CONTEXT ---\n{context_str}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n"
                    f"{user_query}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n"
                )
                
                try:
                    # Initialize LLM call with the token provided via the sidebar
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
                    llm = HuggingFaceEndpoint(
                        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        task="text-generation",
                        temperature=0.1,
                        max_new_tokens=512
                    )
                    
                    llm_response = llm.invoke(llama_prompt)
                    
                    # Render final answer payload to screen
                    st.markdown("### 🎯 System Synthesis Answer")
                    st.write(llm_response.strip())
                    
                    # 4. Loop through hits to check if any visual nodes matched the search criteria
                    referenced_images = [m for m in matched_nodes if m["is_image"]]
                    
                    if referenced_images:
                        st.markdown("### 📊 Extracted Visual Diagrams & Data Elements")
                        
                        # Generate layout columns to display matched images cleanly side-by-side
                        cols = st.columns(len(referenced_images))
                        for idx, img_node in enumerate(referenced_images):
                            with cols[idx]:
                                st.image(
                                    img_node["image_bytes"], 
                                    caption=f"Artifact {idx+1} from source: {img_node['source']}",
                                    use_container_width=True
                                )
                    
                    # Collapsible UI element for inspection and debugging
                    with st.expander("🔍 System Core Inspection: View Raw Matched BM25 Chunks"):
                        for idx, match in enumerate(matched_nodes):
                            st.markdown(f"**Chunk {idx+1} | Source Reference: `{match['source']}` (Is Graphic Node: {match['is_image']})**")
                            st.code(match['text'], language="markdown")
                            st.markdown("---")
                            
                except Exception as e:
                    st.error(f"Inference Engine Connection Timed Out/Failed: {str(e)}")