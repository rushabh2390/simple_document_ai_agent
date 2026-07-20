import os
import sqlite3
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import streamlit as st
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatabaseManager")


class RAGDatabaseManager:
    def __init__(self, db_path: str = "./processed_data/rag_storage.db"):
        """Initializes a clean, high-performance local SQLite storage engine."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.init_db()

    def _get_connection(self):
        """Creates a thread-safe connection to the local SQLite database file."""
        # check_same_thread=False allows Streamlit's multi-threaded UI to query safely
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def init_db(self):
        """Creates standard tables and enables the fast native FTS5 full-text indexing."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 1. Master Table to store layout chunks and visual asset paths
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        filename TEXT,
                        raw_text TEXT,
                        table_path TEXT,
                        image_path TEXT
                    );
                """)

                # 2. Virtual FTS5 Table for lightning-fast BM25 text searches
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS fts5_bm25_idx USING fts5(
                        chunk_id UNINDEXED,
                        text
                    );
                """)
                conn.commit()
                logger.info(
                    "✅ Native SQLite engine & FTS5 full-text indices initialized.")
        except Exception as e:
            logger.critical(
                f"💥 Failed to initialize native database: {str(e)}")

    def insert_document_chunks(self, chunks: List[Dict[str, Any]], filename: str):
        """Inserts a batch of multi-modal document chunks cleanly inside a single transaction."""
        if not chunks:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                for chunk in chunks:
                    # Insert data mappings
                    cursor.execute("""
                        INSERT OR REPLACE INTO document_chunks (chunk_id, filename, raw_text, table_path, image_path)
                        VALUES (?, ?, ?, ?, ?);
                    """, (
                        chunk["chunk_id"],
                        filename,
                        chunk["text"],
                        chunk.get("table_path"),
                        chunk.get("image_path")
                    ))

                    # Mirror raw text into the full-text text search virtual table
                    cursor.execute("""
                        INSERT OR REPLACE INTO fts5_bm25_idx (chunk_id, text) 
                        VALUES (?, ?);
                    """, (chunk["chunk_id"], chunk["text"]))

                conn.commit()
            logger.info(
                f"💾 Successfully indexed {len(chunks)} chunks from '{filename}'.")
        except Exception as e:
            logger.error(f"❌ Failed native SQLite batch insertion: {str(e)}")

    def search_bm25(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Executes full-text keyword searches instantly across thousands of records."""
        # Clean up input string
        clean_words = re.sub(r"[^\w\s]", " ", query.lower()).split()
        if not clean_words:
            return []

        # Standardize search queries for FTS5 syntax
        fts5_query = " OR ".join([f'"{word}"' for word in clean_words])
        like_query = f"%{query}%"

        # FIX: Explicitly match the virtual index table alias inside bm25()
        search_sql = """
            SELECT 
                dc.chunk_id, 
                dc.filename, 
                dc.raw_text, 
                dc.table_path, 
                dc.image_path, 
                bm25(fts5_bm25_idx) AS rank_score
            FROM fts5_bm25_idx idx
            JOIN document_chunks dc ON idx.chunk_id = dc.chunk_id
            WHERE idx.text MATCH ?
            
            UNION ALL
            
            SELECT 
                dc.chunk_id, 
                dc.filename, 
                dc.raw_text, 
                dc.table_path, 
                dc.image_path, 
                999.0 AS rank_score
            FROM document_chunks dc
            WHERE LOWER(dc.raw_text) LIKE LOWER(?)
            
            ORDER BY rank_score ASC
            LIMIT ?;
        """

        results = []
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row  # Access columns natively by name
                cursor = conn.cursor()
                cursor.execute(search_sql, (fts5_query, like_query, limit))
                rows = cursor.fetchall()

                for row in rows:
                    # Prevent duplicates if a node matches both FTS5 and LIKE fallback
                    if any(r["chunk_id"] == row["chunk_id"] for r in results):
                        continue

                    results.append({
                        "chunk_id": row["chunk_id"],
                        "filename": row["filename"],
                        "text": row["raw_text"],
                        "table_path": row["table_path"],
                        "image_path": row["image_path"],
                        # In SQLite BM25, lower scores mean closer matches.
                        # We invert it or present structural metrics clearly here.
                        "score": abs(row["rank_score"]) if row["rank_score"] != 999.0 else 0.0001
                    })
        except Exception as e:
            logger.error(
                f"🔍 Native SQLite search encountered an error: {str(e)}")

        return results

    def wipe_all_data(self):
        """Clears all records instantly, resetting the vector database matrix state."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM document_chunks;")
                cursor.execute("DELETE FROM fts5_bm25_idx;")
                conn.commit()
            logger.info("🗑️ System Data Purge Complete.")
        except Exception as e:
            logger.error(
                f"❌ Error during native tables purge execution: {str(e)}")


@tool
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """
    Searches the uploaded document base (PDFs, text files, code manuals, sales sheets)
    using BM25 keyword routing. Returns relevant text fragments along with document titles.
    """
    configurable = config.get("configurable", {}) if config else {}
    db_manager = configurable.get("db_manager")
    limit = configurable.get("retrieval_limit", 3)
    shared_node_container = configurable.get("shared_node_container")
    
    if not db_manager:
        return "Error: Database manager instance is missing from the agent runtime configuration."

    matched_nodes = db_manager.search_bm25(query, limit=limit)
    if shared_node_container is not None and matched_nodes:
        shared_node_container.extend(matched_nodes)

    if not matched_nodes:
        return "No relevant textual context or metrics found in the knowledge base."

    formatted_chunks = []
    for m in matched_nodes:
        # Start with the basic metadata and row description
        context_chunk = f"[Source File: {m['filename']} | Chunk ID: {m['chunk_id']}]\n{m['text']}"
        
        # Guardrail check for structural table data
        if m.get("table_path") and os.path.exists(m["table_path"]):
            try:
                table_df = pd.read_csv(m["table_path"])
                
                # 🔥 FIX: Only feed the top 3-5 preview rows to the LLM context to prevent token explosion.
                # The full dataset will still load completely in your Streamlit Inspector Panel!
                preview_rows = 5 
                table_markdown = table_df.head(preview_rows).to_markdown(index=False)
                
                context_chunk += (
                    f"\n\n📊 [TABLE PREVIEW (First {preview_rows} rows out of {len(table_df)} total)]:\n{table_markdown}\n"
                    f"Note: The complete table asset has been loaded into the user's Inspector UI layout panel."
                )
            except Exception as e:
                logger.warning(f"Could not safely append table data snippet: {e}")
                
        formatted_chunks.append(context_chunk)

    return "\n\n---\n\n".join(formatted_chunks)
