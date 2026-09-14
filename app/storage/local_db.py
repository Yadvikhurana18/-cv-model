"""
Local SQLite database for offline / fallback mode.
Migrates seamlessly to Supabase when the project comes online.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "argus.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            component_id    TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'NORMAL',
            defect_type     TEXT,
            confidence      REAL,
            camera_id       TEXT,
            notes           TEXT,
            image_path      TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT NOT NULL,
            epochs          INTEGER,
            batch_size      INTEGER,
            learning_rate   REAL,
            accuracy        REAL,
            loss            REAL,
            dataset_path    TEXT,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS components (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            component_id    TEXT UNIQUE NOT NULL,
            name            TEXT,
            category        TEXT,
            status          TEXT DEFAULT 'NORMAL',
            total_inspections INTEGER DEFAULT 0,
            defect_count    INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );
    """)


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_sample():
    with get_db() as conn:
        conn.execute(
            """INSERT INTO inspection_logs
               (component_id, status, defect_type, confidence, camera_id, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("ISRO-CMP-001", "DEFECTIVE", "Surface_Crack", 0.94, "CAM-01",
             "Sample entry — ISRO Component AI Inspection System"),
        )
        conn.execute(
            """INSERT INTO components
               (component_id, name, category, status, total_inspections, defect_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("ISRO-CMP-001", "PCB Mainboard Unit", "Electronics", "DEFECTIVE", 1, 1),
        )
        conn.execute(
            """INSERT INTO training_runs
               (model_name, epochs, batch_size, learning_rate, accuracy, loss, dataset_path, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("mobilenet_v3_large", 20, 32, 0.001, 0.96, 0.12, "dataset/",
             "Initial training run — binary defect classifier"),
        )

    print("[OK] Sample data inserted into local DB:", DB_PATH)
    print()


def query_all(table: str):
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        print(f"\n--- {table} ({len(rows)} rows) ---")
        for row in rows:
            print(dict(row))
        return rows


if __name__ == "__main__":
    insert_sample()
    query_all("inspection_logs")
    query_all("components")
    query_all("training_runs")
