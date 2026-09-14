"""
Database setup script for ISRO Component Inspection System.
Run this ONCE after Supabase project is active:
    python -m app.storage.setup_db
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from supabase import create_client


def setup():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)

    print(f"Connecting to {url} ...")
    client = create_client(url, key)

    # --- Create tables via SQL RPC -------------------------------------------
    sql = """
    -- 1. Inspection logs
    CREATE TABLE IF NOT EXISTS inspection_logs (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        component_id    TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'NORMAL',
        defect_type     TEXT,
        confidence      REAL,
        camera_id       TEXT,
        notes           TEXT,
        image_path      TEXT,
        created_at      TIMESTAMPTZ DEFAULT now()
    );

    -- 2. Training runs
    CREATE TABLE IF NOT EXISTS training_runs (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        model_name      TEXT NOT NULL,
        epochs          INTEGER,
        batch_size      INTEGER,
        learning_rate   REAL,
        accuracy        REAL,
        loss            REAL,
        dataset_path    TEXT,
        notes           TEXT,
        created_at      TIMESTAMPTZ DEFAULT now()
    );

    -- 3. Component registry
    CREATE TABLE IF NOT EXISTS components (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        component_id    TEXT UNIQUE NOT NULL,
        name            TEXT,
        category        TEXT,
        status          TEXT DEFAULT 'NORMAL',
        total_inspections INTEGER DEFAULT 0,
        defect_count    INTEGER DEFAULT 0,
        created_at      TIMESTAMPTZ DEFAULT now()
    );

    -- 4. Enable RLS but allow service_role full access
    ALTER TABLE inspection_logs ENABLE ROW LEVEL SECURITY;
    ALTER TABLE training_runs   ENABLE ROW LEVEL SECURITY;
    ALTER TABLE components      ENABLE ROW LEVEL SECURITY;

    -- Service-role bypasses RLS automatically, but add anon read policy
    CREATE POLICY "anon_read_inspection_logs"
        ON inspection_logs FOR SELECT USING (true);
    CREATE POLICY "anon_read_training_runs"
        ON training_runs   FOR SELECT USING (true);
    CREATE POLICY "anon_read_components"
        ON components      FOR SELECT USING (true);
    """

    try:
        client.rpc("exec_sql", {"query": sql}).execute()
        print("Tables created successfully!")
    except Exception:
        # RPC might not exist — fall back to PostgREST inserts to verify connectivity
        print("Note: exec_sql RPC not available. Testing connection via insert...")

    # --- Insert sample data ---------------------------------------------------
    sample_inspection = {
        "component_id": "ISRO-CMP-001",
        "status": "DEFECTIVE",
        "defect_type": "Surface_Crack",
        "confidence": 0.94,
        "camera_id": "CAM-01",
        "notes": "Sample entry — ISRO Component AI Inspection System",
    }

    sample_component = {
        "component_id": "ISRO-CMP-001",
        "name": "PCB Mainboard Unit",
        "category": "Electronics",
        "status": "DEFECTIVE",
        "total_inspections": 1,
        "defect_count": 1,
    }

    sample_training = {
        "model_name": "mobilenet_v3_large",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "accuracy": 0.96,
        "loss": 0.12,
        "dataset_path": "dataset/",
        "notes": "Initial training run — binary defect classifier",
    }

    results = []

    for table, data in [
        ("inspection_logs", sample_inspection),
        ("components", sample_component),
        ("training_runs", sample_training),
    ]:
        try:
            res = client.table(table).insert(data).execute()
            print(f"[OK] Inserted into {table}: {res.data}")
            results.append((table, True))
        except Exception as e:
            print(f"[FAIL] {table}: {e}")
            results.append((table, False))

    print("\n--- Summary ---")
    for table, ok in results:
        print(f"  {table}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    setup()
