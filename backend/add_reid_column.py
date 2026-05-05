"""
Database migration: add reid_embedding column to tracked_individuals.

Run once from the backend/ directory:
    python add_reid_column.py
"""

import sqlite3
import os
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "violation_tracking.db")


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH} — nothing to migrate.")
        sys.exit(0)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Check if column already exists
    cur.execute("PRAGMA table_info(tracked_individuals)")
    cols = [row[1] for row in cur.fetchall()]

    if "reid_embedding" in cols:
        print("Column 'reid_embedding' already exists — no migration needed.")
        con.close()
        return

    print("Adding 'reid_embedding' column to tracked_individuals ...")
    cur.execute("ALTER TABLE tracked_individuals ADD COLUMN reid_embedding BLOB")
    con.commit()
    con.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
