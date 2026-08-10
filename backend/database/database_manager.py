import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..config.config import logger, settings
from ..database.database import Base, engine

STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


class RAGDatabaseManager:
    def __init__(self):
        """Initializes a clean, high-performance local SQLite storage engine."""
        clean_path = settings.DATABASE_URL.replace("sqlite:///", "")
        self.db_path = Path(clean_path)
        self.init_db()

    def _get_connection(self):
        """Creates a thread-safe connection to the local SQLite database file."""
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def init_db(self):
        """Creates standard tables, FTS5 index, and background ingestion job tracker."""
        try:
            Base.metadata.create_all(bind=engine)
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Master Table: Document Chunks
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        filename TEXT,
                        raw_text TEXT,
                        table_path TEXT,
                        image_path TEXT
                    );
                """)

                # Virtual FTS5 Table: BM25 Full-Text Search
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS fts5_bm25_idx USING fts5(
                        chunk_id UNINDEXED,
                        text
                    );
                """)
                conn.commit()
                logger.info(
                    "✅ Native SQLite engine, FTS5 indices, and Job Status tables initialized."
                )
        except Exception as e:
            logger.critical(f"💥 Failed to initialize native database: {e!s}")

    def insert_document_chunks(self, chunks: list[dict[str, Any]], filename: str):
        """Inserts a batch of multi-modal document chunks cleanly inside a single transaction."""
        if not chunks:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                for chunk in chunks:
                    # Insert data mappings
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO document_chunks (chunk_id, filename, raw_text, table_path, image_path)
                        VALUES (?, ?, ?, ?, ?);
                    """,
                        (
                            chunk["chunk_id"],
                            filename,
                            chunk["text"],
                            chunk.get("table_path"),
                            chunk.get("image_path"),
                        ),
                    )

                    # Mirror raw text into the full-text search virtual table
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO fts5_bm25_idx (chunk_id, text) 
                        VALUES (?, ?);
                    """,
                        (chunk["chunk_id"], chunk["text"]),
                    )

                conn.commit()
            logger.info(
                f"💾 Successfully indexed {len(chunks)} chunks from '{filename}'."
            )
        except Exception as e:
            logger.error(f"❌ Failed native SQLite batch insertion: {e!s}")

    def search_bm25(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Executes full-text keyword searches and formats image/table asset URLs."""
        raw_tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
        search_tokens = [
            w for w in raw_tokens if w not in STOP_WORDS and len(w) > 1
        ] or raw_tokens

        if not search_tokens:
            return []

        fts5_and_query = " AND ".join([f'"{word}"' for word in search_tokens])
        fts5_or_query = " OR ".join([f'"{word}"' for word in search_tokens])

        search_sql = """
            SELECT 
                dc.chunk_id, 
                dc.filename, 
                dc.raw_text, 
                dc.table_path, 
                dc.image_path, 
                bm25(fts5_bm25_idx) AS rank_score
            FROM fts5_bm25_idx
            JOIN document_chunks dc ON fts5_bm25_idx.chunk_id = dc.chunk_id
            WHERE fts5_bm25_idx.text MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?;
        """

        results = []
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(search_sql, (fts5_and_query, limit))
                rows = cursor.fetchall()

                if not rows:
                    cursor.execute(search_sql, (fts5_or_query, limit))
                    rows = cursor.fetchall()

                for row in rows:
                    if any(r["chunk_id"] == row["chunk_id"] for r in results):
                        continue

                    # Format relative/absolute paths into accessible URLs
                    raw_image_path = row["image_path"]
                    raw_table_path = row["table_path"]

                    image_url = (
                        f"{os.path.basename(raw_image_path)}"
                        if raw_image_path
                        else None
                    )
                    table_url = (
                        f"{os.path.basename(raw_table_path)}"
                        if raw_table_path
                        else None
                    )

                    results.append(
                        {
                            "chunk_id": row["chunk_id"],
                            "filename": row["filename"],
                            "text": row["raw_text"],
                            "table_path": raw_table_path,
                            "image_path": raw_image_path,
                            "image_url": image_url,
                            "table_url": table_url,
                            "score": round(abs(float(row["rank_score"])), 4),
                        }
                    )

        except Exception as e:
            logger.error(f"🔍 Native SQLite search encountered an error: {e!s}")

        return results

    def wipe_all_data(self):
        """Clears all records instantly, resetting the database state."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM document_chunks;")
                cursor.execute("DELETE FROM fts5_bm25_idx;")
                conn.commit()
            logger.info("🗑️ System Data Purge Complete.")
        except Exception as e:
            logger.error(f"❌ Error during native tables purge execution: {e!s}")


@tool
def query_tabular_database(sql_query: str, config: RunnableConfig = None) -> str:
    """
    Executes a SQL query against uploaded tabular datasets (CSV/Excel tables) stored in SQLite.
    Use this tool for mathematical aggregations, quarterly/monthly reports, sum calculations,
    counting records, filtering, or finding min/max values in spreadsheets.

    Example: SELECT QTR_ID, SUM(SALES) as total_sales FROM tbl_sales_data_sample GROUP BY QTR_ID
    """
    configurable = config.get("configurable", {}) if config else {}
    shared_node_container = configurable.get("shared_node_container")

    tabular_db_path = Path(settings.STATIC_ASSET_DIR) / "dynamic_tabular_data.db"
    if not tabular_db_path.exists():
        return (
            "Error: No tabular database found. Please upload a CSV or Excel file first."
        )

    try:
        conn_uri = f"file:{tabular_db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(conn_uri, uri=True)

        sql_clean = sql_query.strip().rstrip(";")
        upper_sql = sql_clean.upper()

        if (
            upper_sql.startswith("SELECT")
            and "GROUP BY" not in upper_sql
            and "LIMIT" not in upper_sql
        ):
            sql_clean += " LIMIT 20"

        df_result = pd.read_sql_query(sql_clean, conn)
        conn.close()

        # ✅ POPULATE INSPECTOR PANEL DIRECTLY FOR CSV DATA
        if shared_node_container is not None and not df_result.empty:
            shared_node_container.append(
                {
                    "chunk_id": "sql_query_result",
                    "filename": "dynamic_tabular_data.db",
                    "text": f"SQL Query: {sql_clean}\n\nResult:\n"
                    + df_result.head(10).to_markdown(index=False),
                    "table_path": None,
                    "image_path": None,
                    "score": 1.0,
                }
            )

        if df_result.empty:
            return "Query executed successfully, but returned 0 records."

        return (
            f"SQL Execution Result ({len(df_result)} rows returned):\n\n"
            + df_result.to_markdown(index=False)
        )

    except Exception as e:
        return f"SQL Execution Error: {e!s}. Please check your table and column names in the dataset schema."


# =====================================================================
# TOOL 2: DOCUMENT TEXT RETRIEVAL TOOL (FOR PDF, MD, DOCX)
# =====================================================================
@tool
def search_knowledge_base(query: str, config: RunnableConfig = None) -> str:
    """
    Searches uploaded unstructured documents (PDFs, text files, Word documents)
    using BM25 keyword routing. Returns relevant text passages along with document titles.
    """
    configurable = config.get("configurable", {}) if config else {}
    db_manager = configurable.get("db_manager")
    limit = configurable.get("retrieval_limit", 5)
    shared_node_container = configurable.get("shared_node_container")

    if not db_manager:
        return "Error: Database manager instance is missing from the agent runtime configuration."

    matched_nodes = db_manager.search_bm25(query, limit=limit)
    logger.info("=========================================================")
    logger.info(f"macthing nodes: {matched_nodes}")
    logger.info("=========================================================")
    if shared_node_container is not None and matched_nodes:
        shared_node_container.extend(matched_nodes)

    if not matched_nodes:
        return "No relevant textual context or metrics found in the knowledge base."

    formatted_chunks = []
    total_chars = 0
    MAX_CHAR_CAP = 12000  # Cap around ~3,000 tokens to prevent Ollama context overflow

    for m in matched_nodes:
        context_chunk = (
            f"[Source File: {m['filename']} | Chunk ID: {m['chunk_id']}]\n{m['text']}"
        )

        # Check for structural table previews in PDFs
        if m.get("table_path") and os.path.exists(m["table_path"]):
            try:
                table_df = pd.read_csv(m["table_path"])
                preview_rows = 5
                table_markdown = table_df.head(preview_rows).to_markdown(index=False)

                context_chunk += (
                    f"\n\n📊 [TABLE PREVIEW (First {preview_rows} rows out of {len(table_df)} total)]:\n"
                    f"{table_markdown}\n"
                )
            except Exception as e:
                logger.warning(f"Could not append table data snippet: {e}")

        if total_chars + len(context_chunk) > MAX_CHAR_CAP:
            formatted_chunks.append(
                "\n[Note: Remaining chunks omitted to keep context size safe.]"
            )
            break

        formatted_chunks.append(context_chunk)
        total_chars += len(context_chunk)

    return "\n\n---\n\n".join(formatted_chunks)
