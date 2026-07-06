import os
import logging
from pathlib import Path
from document_parser import MultiModalDocumentParser
from database_manager import RAGDatabaseManager

# Setup basic runtime logging visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAGOrchestrator")


class RAGSystem:
    def __init__(self):
        """Initializes both the parser engine and the persistent SQLite database layer."""
        # Initialize the parser class (handles extraction and threading)
        self.parser = MultiModalDocumentParser()
        
        # Initialize the database manager (handles FTS5 BM25 index and queries)
        self.db = RAGDatabaseManager()

    def ingest_file(self, absolute_file_path: str):
        """
        Reads a file from a full path, extracts multi-modal chunks,
        and saves everything into the persistent SQLite index.
        """
        path = Path(absolute_file_path)
        if not path.exists():
            logger.error(f"❌ Target file not found at: '{absolute_file_path}'")
            return False

        filename = path.name
        logger.info(f"📂 Reading file bytes into buffer: {filename}")
        
        try:
            # 1. Read file as bytes (simulating a Streamlit upload widget buffer)
            with open(path, "rb") as f:
                file_bytes = f.read()

            # 2. Extract layout-mapped text, tables, and images
            extracted_chunks = self.parser.parse_file(file_bytes=file_bytes, filename=filename)

            if not extracted_chunks:
                logger.warning(f"⚠️ No structural content extracted from '{filename}'.")
                return False

            # 3. Store chunks and multi-modal asset references to the DB
            self.db.insert_document_chunks(chunks=extracted_chunks, filename=filename)
            print(f"\n🎉 SUCCESS: Successfully indexed '{filename}' ({len(extracted_chunks)} chunks).")
            return True

        except Exception as e:
            logger.error(f"💥 Failed to complete ingestion pipeline for '{filename}': {str(e)}", exc_info=True)
            return False

    def search(self, query: str, top_n: int = 3):
        """
        Performs a local native BM25 search and returns results formatted
        with their associated text, images, and tables.
        """
        print("\n" + "="*60)
        print(f"🔍 RETRIEVAL SEARCH FOR QUERY: '{query}'")
        print("="*60)

        matches = self.db.search_bm25(query=query, limit=top_n)

        if not matches:
            print("⚠️ No matching chunks found in the index database.")
            return

        for idx, match in enumerate(matches):
            print(f"\n[MATCH {idx+1}] (Score: {match['score']:.4f}) From File: {match['filename']}")
            print(f"📄 TEXT SNIPPET:\n{match['text']}")
            
            # Highlight linked multi-modal components if present
            if match['table_path']:
                print(f"📊 ASSOCIATED EXPORTED TABLE (CSV): {match['table_path']}")
            if match['image_path']:
                print(f"🖼️ ASSOCIATED EXPORTED IMAGE (PNG): {match['image_path']}")
                
            print("-" * 50)

    def clear_system(self):
        """Purges the database index files."""
        self.db.wipe_all_data()


# =====================================================================
# RUNNER EXECUTION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    # 1. Initialize the system components
    rag_engine = RAGSystem()

    # [OPTIONAL] Uncomment the line below if you want to wipe previous runs before starting
    # rag_engine.clear_system()

    # 2. Target file definition (Provide your actual target file path here)
    # This architecture accepts PDF, DOCX, XLSX, Markdown, or Plain Text
    target_document = r"E:\Deep-Learning-with-PyTorch.pdf"

    if os.path.exists(target_document):
        # 3. Process layout extraction and write to DB
        # rag_engine.ingest_file(absolute_file_path=target_document)

        # 4. Perform a sample query evaluation
        # Try looking for a topic that has prominent diagrams or tables in your text book
        # sample_query = "alexnet convolutional networks layer design specifications"
        sample_query = "Explain tensor with example from the book"
        rag_engine.search(query=sample_query, top_n=2)
        
    else:
        print(f"❌ Test aborted: Please update 'target_document' to point to a valid file path.")