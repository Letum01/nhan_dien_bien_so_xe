"""Small SQLite persistence layer for opt-in ALPR recognition history."""

from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


class RecognitionDatabase:
    """SQLite history store. Connections are scoped to each operation."""

    def __init__(self, database_path: str = "data/recognitions.sqlite3") -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                processed_text TEXT NOT NULL,
                recognized_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL,
                status TEXT NOT NULL,
                image_path TEXT
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_recognitions_time ON recognitions(recognized_at)")

    def add(self, raw_text: str, processed_text: str, model: str, source: str,
            confidence: Optional[float] = None, status: str = "recognized",
            image_path: Optional[str] = None) -> int:
        """Insert a result using bound parameters and return its row ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO recognitions
                (raw_text, processed_text, model, source, confidence, status, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (raw_text, processed_text, model, source, confidence, status, image_path),
            )
            return int(cursor.lastrowid)

    def search(self, query: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """Search text fields and return newest records first."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM recognitions
                WHERE processed_text LIKE ? OR raw_text LIKE ?
                ORDER BY recognized_at DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", max(1, min(int(limit), 10000))),
            ).fetchall()
            return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM recognitions").fetchone()[0])

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM recognitions")