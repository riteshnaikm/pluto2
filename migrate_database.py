"""
Database Migration Script - Add JobID Feature
Run this once to update your database schema
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

def migrate_database():
    """Add oorwin_job_id column and create recruiter_handbooks table"""
    conn = sqlite3.connect('combined_db.db')
    cursor = conn.cursor()
    
    print("Starting database migration...")
    
    # 1. Add oorwin_job_id column to evaluations table
    try:
        cursor.execute('''
            ALTER TABLE evaluations 
            ADD COLUMN oorwin_job_id TEXT;
        ''')
        print("✅ Added oorwin_job_id column to evaluations table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("ℹ️  Column oorwin_job_id already exists in evaluations table")
        else:
            print(f"❌ Error adding column to evaluations: {e}")
    
    # 2. Create recruiter_handbooks table
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recruiter_handbooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                oorwin_job_id TEXT,
                job_title TEXT,
                job_description TEXT,
                additional_context TEXT,
                markdown_content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Created recruiter_handbooks table")
    except sqlite3.OperationalError as e:
        print(f"ℹ️  recruiter_handbooks table already exists: {e}")
    
    # 3. Create indexes for faster queries
    try:
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_evaluations_job_id 
            ON evaluations(oorwin_job_id)
        ''')
        print("✅ Created index on evaluations.oorwin_job_id")
    except sqlite3.OperationalError as e:
        print(f"ℹ️  Index already exists: {e}")
    
    try:
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_handbooks_job_id 
            ON recruiter_handbooks(oorwin_job_id)
        ''')
        print("✅ Created index on recruiter_handbooks.oorwin_job_id")
    except sqlite3.OperationalError as e:
        print(f"ℹ️  Index already exists: {e}")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print("\n✅ Database migration completed successfully!")
    print("You can now restart your application.")

if __name__ == "__main__":
    migrate_database()

