import sqlite3

db_path = r"processed_data\rag_storage.db"

# 2. Open a completely isolated connection directly to the file
conn = sqlite3.connect(db_path, isolation_level=None)  # Auto-commit mode
cursor = conn.cursor()

try:
    # Disable foreign key locks temporarily
    cursor.execute("PRAGMA foreign_keys = OFF;")

    # Delete from FTS virtual index (The exact query that worked in DBeaver)
    cursor.execute("DELETE FROM fts5_bm25_idx;")

    # Delete from main document tables
    cursor.execute("DELETE FROM document_chunks;")
    cursor.execute("DELETE FROM ingestion_jobs;")

    cursor.execute("VACUUM;")

    print("🗑️ Data successfully purged via isolated connection.")

except Exception as e:
    print(f"❌ Error during isolated purge: {e!s}")
    raise e
finally:
    # Close connection immediately so no locks remain
    cursor.close()
    conn.close()
