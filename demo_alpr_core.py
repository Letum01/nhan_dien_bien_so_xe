"""Bounded image input and isolated, opt-in demo history."""

import csv
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from io import StringIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from final_ocr_intake import preview_bytes


def decode(data: bytes) -> np.ndarray:
    """Validate upload limits and decode to the existing OpenCV BGR contract."""
    preview_bytes(data)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không đọc được ảnh JPG/PNG')
    return image


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Produce Excel-compatible UTF-8 CSV, guarding spreadsheet formulas."""
    if not rows:
        return b''

    def safe(value: Any) -> Any:
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n')):
            return "'" + value
        return value

    out = StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows({key: safe(value) for key, value in row.items()} for row in rows)
    return out.getvalue().encode('utf-8-sig')


class History:
    """Own one short-lived connection per operation; never store pixels."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY,
                filename TEXT, prediction TEXT, algorithm TEXT, seconds REAL, status TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, rows: Iterable[Mapping[str, Any]]) -> None:
        with closing(self.connect()) as db, db:
            db.executemany('''INSERT INTO history(filename,prediction,algorithm,seconds,status)
                              VALUES (:filename,:prediction,:algorithm,:seconds,:status)''', rows)

    def search(self, query: str = '') -> list[dict]:
        with closing(self.connect()) as db:
            rows = db.execute('''SELECT * FROM history WHERE instr(lower(filename),lower(?))
                                 OR instr(lower(prediction),lower(?)) ORDER BY id DESC''', (query, query))
            return [dict(row) for row in rows]

    def clear(self) -> None:
        with closing(self.connect()) as db, db:
            db.execute('DELETE FROM history')