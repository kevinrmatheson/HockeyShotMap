import sqlite3
import os

db_path = 'hockey_data.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names (excluding sqlite_sequence which is internal)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
    tables = cursor.fetchall()
    
    print(f'Found {len(tables)} tables to drop:')
    for table in tables:
        print(f'  - {table[0]}')
    
    # Drop all tables
    for table in tables:
        cursor.execute(f'DROP TABLE IF EXISTS {table[0]}')
    
    conn.commit()
    conn.close()
    print('All tables dropped successfully!')
else:
    print(f'Database {db_path} does not exist')