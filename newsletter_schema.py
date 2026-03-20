import sqlite3

DB_PATH = "newsletter_agent.db"


def ensure_column(cursor, table_name, column_name, column_definition):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}
    if column_name not in columns:
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def initialize_database(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS newsletter_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief TEXT NOT NULL,
            depth TEXT,
            explanation_style TEXT,
            custom_style_instructions TEXT,
            audience TEXT,
            tone TEXT,
            title TEXT,
            queries_json TEXT NOT NULL,
            sections_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            output_path TEXT
        )
        """
    )

    ensure_column(cursor, "newsletter_runs", "depth", "TEXT")
    ensure_column(cursor, "newsletter_runs", "explanation_style", "TEXT")
    ensure_column(cursor, "newsletter_runs", "custom_style_instructions", "TEXT")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS source_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            rank_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            snippet TEXT,
            article_text TEXT,
            source_summary TEXT,
            relevance_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, url),
            FOREIGN KEY (run_id) REFERENCES newsletter_runs (id)
        )
        """
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
    print("Newsletter database initialized.")
