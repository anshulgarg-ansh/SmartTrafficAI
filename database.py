
import sqlite3
from pathlib import Path

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "database" / "smarttraffic.db"


def get_db():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ==========================================
# INITIALIZE DATABASE
# ==========================================

def init_db():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_db()

    cursor = conn.cursor()

    # ======================================
    # VEHICLES TABLE
    # ======================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            vehicle_number TEXT UNIQUE NOT NULL,

            owner_name TEXT NOT NULL,

            phone TEXT,

            vehicle_type TEXT,

            insurance_status TEXT DEFAULT 'Unknown',

            registration_status TEXT DEFAULT 'Active'

        )
    """)

    # ======================================
    # VIOLATIONS TABLE
    # ======================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            vehicle_number TEXT NOT NULL,

            violation_type TEXT NOT NULL,

            location TEXT,

            violation_time TEXT NOT NULL,

            evidence_path TEXT,

            penalty REAL DEFAULT 0,

            status TEXT DEFAULT 'Pending'

        )
    """)

    # ======================================
    # AI DETECTION EVENTS TABLE
    # ======================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_events (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            event_key TEXT UNIQUE NOT NULL,

            vehicle_number TEXT,

            violation_type TEXT NOT NULL,

            location TEXT DEFAULT 'Unknown',

            detected_at TEXT NOT NULL,

            confidence REAL DEFAULT 0,

            plate_confidence REAL DEFAULT 0,

            evidence_path TEXT,

            source_video TEXT,

            review_status TEXT DEFAULT 'Pending Review'
                CHECK (
                    review_status IN (
                        'Pending Review',
                        'Approved',
                        'Rejected'
                    )
                ),

            violation_id INTEGER,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (violation_id)
                REFERENCES violations(id)

        )
    """)

    # ======================================
    # DATABASE INDEXES
    # ======================================

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_detection_review
        ON detection_events(review_status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_detection_vehicle
        ON detection_events(vehicle_number)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_violations_vehicle
        ON violations(vehicle_number)
    """)

    # ======================================
    # SAMPLE VEHICLE RECORDS
    # ======================================

    cursor.execute("""
        INSERT OR IGNORE INTO vehicles
        (
            vehicle_number,
            owner_name,
            phone,
            vehicle_type,
            insurance_status
        )
        VALUES

        (
            'MP09AB1234',
            'Rahul Sharma',
            '9876543210',
            'Bike',
            'Valid'
        ),

        (
            'MP09CD5678',
            'Amit Verma',
            '9876543211',
            'Car',
            'Expired'
        ),

        (
            'MP09EF9012',
            'Priya Singh',
            '9876543212',
            'Bike',
            'Valid'
        )
    """)

    conn.commit()

    conn.close()


# ==========================================
# DATABASE TEST
# ==========================================

if __name__ == "__main__":

    init_db()

    print("SmartTraffic database initialized successfully!")

    conn = get_db()

    tables = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """).fetchall()

    print("\nAvailable Database Tables:")

    for table in tables:
        print("-", table["name"])

    conn.close()
