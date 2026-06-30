import sqlite3
from pathlib import Path
import datetime

DEFAULT_DB_PATH = "data/patient_records.db"

def get_connection(db_path=DEFAULT_DB_PATH):
    """
    Returns a connection to the SQLite database, ensuring directory exists.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db(db_path=DEFAULT_DB_PATH):
    """
    Initializes the patient and scan tables in the database.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Scans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prediction_score REAL NOT NULL,
                prediction_label TEXT NOT NULL,
                gradcam_path TEXT,
                doctor_notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        
        conn.commit()
    print(f"Database initialized at {db_path}")

def add_patient(patient_id, name, age, gender, db_path=DEFAULT_DB_PATH):
    """
    Adds a new patient to the database. If they exist, ignores/updates their profile.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO patients (patient_id, name, age, gender)
            VALUES (?, ?, ?, ?)
        """, (patient_id, name, age, gender))
        conn.commit()
    return patient_id

def add_scan(patient_id, image_path, prediction_score, prediction_label, gradcam_path=None, doctor_notes="", db_path=DEFAULT_DB_PATH):
    """
    Logs a new scan prediction to the database.
    """
    scan_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scans (patient_id, scan_date, image_path, prediction_score, prediction_label, gradcam_path, doctor_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, scan_date, image_path, prediction_score, prediction_label, gradcam_path, doctor_notes))
        conn.commit()
        scan_id = cursor.lastrowid
    return scan_id

def get_patient_profile(patient_id, db_path=DEFAULT_DB_PATH):
    """
    Retrieves demographic info for a single patient.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_patient_scans(patient_id, db_path=DEFAULT_DB_PATH):
    """
    Retrieves all scan records for a single patient, sorted by date (newest first).
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY scan_date DESC", (patient_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_all_scans(db_path=DEFAULT_DB_PATH):
    """
    Retrieves all scan records joined with patient names.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.patient_id, p.name, p.age, p.gender, s.scan_date, 
                   s.image_path, s.prediction_score, s.prediction_label, 
                   s.gradcam_path, s.doctor_notes
            FROM scans s
            JOIN patients p ON s.patient_id = p.patient_id
            ORDER BY s.scan_date DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def update_doctor_notes(scan_id, notes, db_path=DEFAULT_DB_PATH):
    """
    Updates the diagnosis or medical notes for a specific scan.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE scans SET doctor_notes = ? WHERE id = ?", (notes, scan_id))
        conn.commit()
    return True

if __name__ == "__main__":
    init_db()
    # Test data
    add_patient("P100", "John Doe", 45, "Male")
    add_scan("P100", "data/test.png", 0.82, "Tuberculosis", doctor_notes="Check validation")
    print(get_patient_scans("P100"))
