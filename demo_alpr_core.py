"""Bounded image input and isolated, opt-in demo history."""
import csv
from io import StringIO
from contextlib import closing
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from final_ocr_intake import preview_bytes


def decode(data):
    preview_bytes(data)
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Không đọc được ảnh JPG/PNG')
    return image


def csv_bytes(rows):
    if not rows:return b''
    def safe(value):
        return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r','\n')) else value
    out=StringIO(newline=''); writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader()
    writer.writerows({k:safe(v) for k,v in r.items()} for r in rows)
    return out.getvalue().encode('utf-8-sig')


class History:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY,
                filename TEXT, prediction TEXT, algorithm TEXT, seconds REAL, status TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')

    def connect(self):
        con=sqlite3.connect(self.path,timeout=10);con.row_factory=sqlite3.Row
        return con

    def save(self,rows):
        with closing(self.connect()) as db, db:
            db.executemany('INSERT INTO history(filename,prediction,algorithm,seconds,status) VALUES (:filename,:prediction,:algorithm,:seconds,:status)',rows)

    def search(self,query=''):
        with closing(self.connect()) as db:
            return [dict(r) for r in db.execute('SELECT * FROM history WHERE instr(lower(filename),lower(?)) OR instr(lower(prediction),lower(?)) ORDER BY id DESC',(query,query))]

    def clear(self):
        with closing(self.connect()) as db, db:db.execute('DELETE FROM history')