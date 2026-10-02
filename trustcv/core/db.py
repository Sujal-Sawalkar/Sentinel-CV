import sqlite3
from pathlib import Path

def connect(db_path):
    return sqlite3.connect(db_path)

def init_db(db_path):
    con = connect(db_path)
    cur = con.cursor()
    
    # Models registry table
    cur.execute("""CREATE TABLE IF NOT EXISTS models(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        version TEXT DEFAULT '1.0',
        model_type TEXT DEFAULT 'Computer Vision',
        status TEXT DEFAULT 'ACTIVE',
        file_size INTEGER DEFAULT 0,
        registered_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    
    # Check if existing models table needs columns
    cur.execute("PRAGMA table_info(models)")
    existing_cols = [row[1] for row in cur.fetchall()]
    if "version" not in existing_cols:
        try: cur.execute("ALTER TABLE models ADD COLUMN version TEXT DEFAULT '1.0'")
        except Exception: pass
    if "model_type" not in existing_cols:
        try: cur.execute("ALTER TABLE models ADD COLUMN model_type TEXT DEFAULT 'Computer Vision'")
        except Exception: pass
    if "status" not in existing_cols:
        try: cur.execute("ALTER TABLE models ADD COLUMN status TEXT DEFAULT 'ACTIVE'")
        except Exception: pass
    if "file_size" not in existing_cols:
        try: cur.execute("ALTER TABLE models ADD COLUMN file_size INTEGER DEFAULT 0")
        except Exception: pass

    # Cryptographic hash-chained audit log
    cur.execute("""CREATE TABLE IF NOT EXISTS audit(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        status TEXT NOT NULL,
        details TEXT NOT NULL,
        previous_hash TEXT,
        record_hash TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    # Inference certificate ledger index
    cur.execute("""CREATE TABLE IF NOT EXISTS certificates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        inference_id TEXT UNIQUE NOT NULL,
        timestamp_utc TEXT NOT NULL,
        input_sha256 TEXT NOT NULL,
        model_sha256 TEXT NOT NULL,
        output_preview TEXT,
        status TEXT DEFAULT 'VALID',
        certificate_path TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    con.commit()
    con.close()
