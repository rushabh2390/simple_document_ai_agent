# Multimodal RAG Hub: Document Layout Parsing & Image Retrieval Engine

An enterprise-grade, local-first Multimodal Retrieval-Augmented Generation (RAG) system built with **IBM's Docling v2**, **BM25 exact match indexing**, and **Hugging Face cloud inference**. 

This system breaks down the traditional "unstructured data problem" by visually parsing complex document structures (PDFs, Word files, Excel spreadsheets, ZIP archives) into clean Markdown, isolating embedded visual elements (charts, tables, diagrams), and tracking them cleanly in an in-memory lexical keyword index.

---

## 🚀 Architectural Advantages

1. **Vision-Driven Layout Analysis**: Leverages `Docling`'s deep-learning computer vision engines to preserve reading order, multi-column articles, and nested structures.
2. **Deterministic Table & Image Extraction**: Extracts embedded figures, diagrams, and complex financial matrices directly from byte streams, assigning text-based semantic caption weights to them.
3. **Lexical Retrieval Engine (BM25)**: Bypasses dense vector database server overhead and memory limitations ($TF-IDF$ Term-Frequency constraints) by storing inverted keyword indices natively in system volatile RAM.
4. **Multimodal State Matching**: When a keyword query points to data inside a chart, the BM25 engine surfaces the descriptive context placeholder, streaming the exact visual image buffer straight to your user UI layout.

---

## 🛠️ System Design Topology

* **Frontend Dashboard**: Streamlit (Python Interactive Web Engine Layout).
* **Ingestion Layer**: In-Memory Byte Decompression (`zipfile`, `io.BytesIO`) combined with `Docling v2 HybridChunker`.
* **Search Core**: `rank_bm25` (Statistical Exact Term Frequency Matching).
* **Inference Model**: Meta `Llama-3-8B-Instruct` (Hosted on Hugging Face Serverless API Hub).

---

## 📥 Prerequisites & Environment Setup

Ensure you have Python 3.10 or greater installed locally on your system.

### 1. Clone the Repository Workspace
```bash
git clone [https://github.com/yourusername/multimodal-rag-vault.git](https://github.com/yourusername/multimodal-rag-vault.git)
cd multimodal-rag-vault
```
### 2. Set Up a Python Virtual Environment
``` bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies.
Install Pre-requistics requirement
```bash
pip install -r requirments.txt
```

### 4. 💻 Run Execution Configuration
``` bash
streamlit run app_ui.py
```


📖 Operational Guide
Authentication Access: Paste your personal Hugging Face User Access Token (hf_...) into the designated masked password input box located inside the left sidebar panel menu.

Chunk Constraint Budgeting: Adjust the Max Target Tokens Per Chunk slider (default: 400 tokens). This uses Docling's HybridChunker token-boundary engine to prevent structural sentences or data cells from getting clipped mid-string.

Ingest Documentation Data:

Drop individual files directly (.pdf, .docx, .xlsx, .csv, .txt, .md).

Alternatively, drop a single compressed .zip bundle. The pipeline unzips the files strictly in memory and iterates over the inner contents automatically.

Interact & Debug: Submit inquiries via the main prompt line. If search criteria variables match text written inside or near a PDF corporate diagram, the engine displays the Llama-3 synthesis text response and renders the underlying extracted charts side-by-side on your dashboard. Expand the system debugging toggle to review exact algorithmic scores.

⚠️ Important Implementation Notes: Troubleshooting Docling v2 Moving Imports
If your system displays an ImportError on initial startup regarding PdfPipelineOptions, verify that your app_ui.py matches the updated Docling v2 core specification guidelines where the structural configuration namespaces have been decoupled from the primary layout engine:

``` bash
# ❌ DEPRECATED IN V1.X (Will throw terminal errors)
# from docling.document_converter import DocumentConverter, PdfPipelineOptions

#   CORRECT V2.X SYSTEM CONFIGURATION BLUEPRINT
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions
from docling.datamodel.base_models import InputFormat

# Instantiate configuration values dynamically via internal property dots
pipeline_options = PipelineOptions()
pipeline_options.pdf_options.images_scale = 1.0  # Force exact image capture scaling

```

🤝 Contribution Guidelines
For architectural enhancements, pull requests, or database state tracking modifications, please open an issue thread detailing your optimization proposal.